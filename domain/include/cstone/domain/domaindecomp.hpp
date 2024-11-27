/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Functions to assign a global cornerstone octree to different ranks
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * Any code in this file relies on a global cornerstone octree on each calling rank.
 */

#pragma once

#include <algorithm>
#include <numeric>
#include <vector>

#include "cstone/tree/csarray.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/util/gsl-lite.hpp"
#include "index_ranges.hpp"

namespace cstone
{

//! @brief determine bins that produce a histogram with uniform number of elements
template<class IndexType>
void uniformBins(const std::vector<IndexType>& counts, gsl::span<TreeNodeIndex> bins, gsl::span<LocalIndex> binCounts)
{
    std::vector<uint64_t> countScan(counts.size() + 1, 0);
    std::inclusive_scan(counts.begin(), counts.end(), countScan.begin() + 1, std::plus<>{}, uint64_t(0));

    int numBins   = bins.size() - 1;
    auto binCount = double(countScan.back()) / numBins;

    bins.front() = 0;
    bins.back()  = counts.size();
#pragma omp parallel for
    for (int i = 1; i < numBins; ++i)
    {
        uint64_t targetCount = i * binCount;
        bins[i]              = std::lower_bound(countScan.begin(), countScan.end(), targetCount) - countScan.begin();
    }
    for (int i = 1; i < numBins; ++i)
    {
        binCounts[i - 1] = countScan[bins[i]] - countScan[bins[i - 1]];
    }
    binCounts.back() = countScan.back() - countScan[bins[numBins - 1]];
}

//! @brief Stores which parts of the SFC belong to which rank. Each rank as an identical copy
template<class KeyType>
class SfcAssignment
{
public:
    SfcAssignment()
        : rankBoundaries_(1)
    {
    }

    explicit SfcAssignment(int numRanks)
        : rankBoundaries_(numRanks + 1)
        , counts_(numRanks)
    {
    }

    KeyType* data() { return rankBoundaries_.data(); }
    const KeyType* data() const { return rankBoundaries_.data(); }

    unsigned* counts() { return counts_.data(); }

    void set(int rank, KeyType a, LocalIndex count)
    {
        rankBoundaries_[rank] = a;
        if (rank < int(counts_.size())) { counts_[rank] = count; }
    }

    [[nodiscard]] int numRanks() const { return int(rankBoundaries_.size()) - 1; }
    [[nodiscard]] KeyType operator[](int rank) const { return rankBoundaries_[rank]; }
    [[nodiscard]] LocalIndex totalCount(int rank) const { return counts_[rank]; }

    [[nodiscard]] int findRank(KeyType key) const
    {
        auto it = std::upper_bound(begin(rankBoundaries_), end(rankBoundaries_), key);
        return int(it - begin(rankBoundaries_)) - 1;
    }

private:
    std::vector<KeyType> rankBoundaries_;
    std::vector<LocalIndex> counts_;
};

template<class KeyType>
SfcAssignment<KeyType> makeSfcAssignment(int numRanks, const std::vector<unsigned>& counts, const KeyType* tree)
{
    SfcAssignment<KeyType> ret(numRanks);
    std::vector<TreeNodeIndex> nodeBins(numRanks + 1);
    uniformBins(counts, nodeBins, {ret.counts(), size_t(numRanks)});
    gather(gsl::span<const TreeNodeIndex>{nodeBins.data(), nodeBins.size()}, tree, ret.data());

    return ret;
}

/*! @brief limit SFC range assignment transfer to the domain of the rank above or below
 *
 * @tparam        KeyType          32- or 64-bit unsigned integer
 * @param[in]     oldAssignment    SFC key assignment boundaries to ranks from the previous step
 * @param[inout]  newAssignment    the current assignment, will be modified if needed
 * @param[in]     tree             the global octree leaves used for domain decomposition in the current step
 * @param[in]     counts           particle counts per leaf cell in @p newTree
 *
 * When assignment boundaries change, we limit the growth of any rank downwards or upwards the SFC
 * to the previous assignment of the rank below or above, i.e. rank r can only acquire new SFC areas
 * that belonged to ranks r-1 or r+1 in the previous step. Only required in extreme cases or testing scenarios
 * to guarantee that the in-focus LET resolution is never exceeded in the trees of other ranks.
 */
template<class KeyType>
void limitBoundaryShifts(const SfcAssignment<KeyType> oldAssignment,
                         SfcAssignment<KeyType>& newAssignment,
                         gsl::span<const KeyType> tree,
                         gsl::span<const unsigned> counts)
{
    int numRanks = std::min(oldAssignment.numRanks(), newAssignment.numRanks()); // oldAssignment empty on first call

    bool triggerRecount = false;
    for (int rank = 1; rank < numRanks; ++rank)
    {
        KeyType newBoundary = std::min(std::max(newAssignment[rank], oldAssignment[rank - 1]), oldAssignment[rank + 1]);
        if (newBoundary != newAssignment[rank])
        {
            triggerRecount             = true;
            newAssignment.data()[rank] = newBoundary;
        }
    }
    if (!triggerRecount) { return; }

    for (int rank = 0; rank < numRanks; ++rank)
    {
        auto a                       = findNodeAbove(tree.data(), nNodes(tree), newAssignment[rank]);
        auto b                       = findNodeAbove(tree.data(), nNodes(tree), newAssignment[rank + 1]);
        std::size_t rankCount        = std::accumulate(counts.begin() + a, counts.begin() + b, std::size_t(0));
        newAssignment.counts()[rank] = rankCount;
    }
}

/*! @brief translates an assignment of a given tree to a new tree
 *
 * @tparam     KeyType         32- or 64-bit unsigned integer
 * @param[in]  assignment      domain assignment
 * @param[in]  focusTree       focus tree leaves
 * @param[in]  peerRanks       list of peer ranks
 * @param[in]  myRank          executing rank ID
 * @param[out] focusAssignment assignment with the same SFC key ranges per
 *                             peer rank as the domain @p assignment,
 *                             but with indices valid w.r.t @p focusTree
 *
 * The focus assignment is implemented as a plain vector; since only
 * the ranges of peer ranks (and not all ranks) are set, the requirements
 * of SpaceCurveAssignment are not met and its findRank() function would not work.
 */
template<class KeyType>
void translateAssignment(const SfcAssignment<KeyType>& assignment,
                         gsl::span<const KeyType> focusTree,
                         gsl::span<const int> peerRanks,
                         int myRank,
                         std::vector<TreeIndexPair>& focusAssignment)
{
    focusAssignment.resize(assignment.numRanks());
    std::fill(focusAssignment.begin(), focusAssignment.end(), TreeIndexPair(0, 0));
    for (int peer : peerRanks)
    {
        // Note: start-end range is narrowed down if no exact match is found.
        // the discarded part will not participate in peer/halo exchanges
        TreeNodeIndex startIndex = findNodeAbove(focusTree.data(), focusTree.size(), assignment[peer]);
        TreeNodeIndex endIndex   = findNodeBelow(focusTree.data(), focusTree.size(), assignment[peer + 1]);

        if (endIndex < startIndex) { endIndex = startIndex; }
        focusAssignment[peer] = TreeIndexPair(startIndex, endIndex);
    }

    TreeNodeIndex newStartIndex = findNodeAbove(focusTree.data(), focusTree.size(), assignment[myRank]);
    TreeNodeIndex newEndIndex   = findNodeBelow(focusTree.data(), focusTree.size(), assignment[myRank + 1]);
    focusAssignment[myRank]     = TreeIndexPair(newStartIndex, newEndIndex);
}

/*! @brief Based on global assignment, create the list of local particle index ranges to send to each rank
 *
 * @tparam KeyType      32- or 64-bit integer
 * @param assignment    global space curve assignment to ranks
 * @param particleKeys  sorted list of SFC keys of local particles present on this rank
 * @return              for each rank, a list of index ranges into @p particleKeys to send
 *
 * Converts the global assignment particle keys ranges into particle indices with binary search
 */
template<class KeyType>
SendRanges createSendRanges(const SfcAssignment<KeyType>& assignment, gsl::span<const KeyType> particleKeys)
{
    int numRanks = assignment.numRanks();

    SendRanges ret(numRanks + 1);
    for (int rank = 0; rank <= numRanks; ++rank)
    {
        KeyType rangeStart = assignment[rank];
        ret[rank] = std::lower_bound(particleKeys.begin(), particleKeys.end(), rangeStart) - particleKeys.begin();
    }

    return ret;
}

/*! @brief return @p numRanks equal length SFC segments for initial domain decomposition
 *
 * @tparam KeyType
 * @param numRanks    number of segments
 * @param level       maximum tree depths or (=number of non-zero leading octal digits)
 * @return            the segments
 *
 * Example: returns [0 2525200000 5252500000 10000000000] for numRanks = 3 and level = 5
 */
template<class KeyType>
std::vector<KeyType> initialDomainSplits(int numRanks, int level)
{
    std::vector<KeyType> ret(numRanks + 1);
    KeyType delta = nodeRange<KeyType>(0) / numRanks;

    ret.front() = 0;
    for (int i = 1; i < numRanks; ++i)
    {
        ret[i] = enclosingBoxCode(KeyType(i) * delta, level);
    }
    ret.back() = nodeRange<KeyType>(0);

    return ret;
}

} // namespace cstone
