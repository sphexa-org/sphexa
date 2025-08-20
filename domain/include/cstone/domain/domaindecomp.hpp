/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
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
#include <span>
#include <vector>

#include "cstone/tree/csarray.hpp"
#include "cstone/primitives/gather.hpp"
#include "index_ranges.hpp"

namespace cstone
{

//! @brief determine bins that produce a histogram with uniform number of elements
template<class IndexType>
void uniformBins(const std::vector<IndexType>& counts, std::span<TreeNodeIndex> bins, std::span<LocalIndex> binCounts)
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
        , numNodesPerRank_(numRanks)
        , treeOffsets_(numRanks + 1)
    {
    }

    KeyType* data() { return rankBoundaries_.data(); }
    const KeyType* data() const { return rankBoundaries_.data(); }

    std::span<LocalIndex> counts() { return counts_; }

    std::span<TreeNodeIndex> numNodesPerRank() { return numNodesPerRank_; }
    std::span<const TreeNodeIndex> numNodesPerRankConst() const { return numNodesPerRank_; }

    std::span<TreeNodeIndex> treeOffsets() { return treeOffsets_; }
    std::span<const TreeNodeIndex> treeOffsetsConst() const { return treeOffsets_; }

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

    //! number of assigned global tree nodes for each rank
    std::vector<TreeNodeIndex> numNodesPerRank_;
    //! scan of numTreeNodes
    std::vector<TreeNodeIndex> treeOffsets_;
};

template<class KeyType>
SfcAssignment<KeyType> makeSfcAssignment(int numRanks, const std::vector<unsigned>& counts, const KeyType* tree)
{
    SfcAssignment<KeyType> ret(numRanks);
    uniformBins(counts, ret.treeOffsets(), ret.counts());
    gather(ret.treeOffsetsConst(), tree, ret.data());

    std::span numNodesPerRank = ret.numNodesPerRank();
    std::span offsets         = ret.treeOffsets();
    for (TreeNodeIndex i = 0; i < numRanks; ++i)
    {
        numNodesPerRank[i] = offsets[i + 1] - offsets[i];
    }

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
                         std::span<const KeyType> tree,
                         std::span<const unsigned> counts)
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

    std::span treeOffsets = newAssignment.treeOffsets();
    treeOffsets.front()   = 0;
    treeOffsets.back()    = nNodes(tree);

    std::span numNodesPerRank = newAssignment.numNodesPerRank();
    for (int rank = 1; rank < numRanks; ++rank)
    {
        treeOffsets[rank]         = findNodeAbove(tree.data(), nNodes(tree), newAssignment[rank]);
        numNodesPerRank[rank - 1] = treeOffsets[rank] - treeOffsets[rank - 1];
    }
    numNodesPerRank[numRanks - 1] = treeOffsets.back() - treeOffsets[numRanks - 1];

    std::span newCounts = newAssignment.counts();
    for (int rank = 0; rank < numRanks; ++rank)
    {
        newCounts[rank] =
            std::accumulate(counts.begin() + treeOffsets[rank], counts.begin() + treeOffsets[rank + 1], std::size_t(0));
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
                         std::span<const KeyType> focusTree,
                         std::span<const int> peerRanks,
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
SendRanges createSendRanges(const SfcAssignment<KeyType>& assignment, std::span<const KeyType> particleKeys)
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

//! @brief translate send ranges indexed in o1 ordering to o2 ordering
inline SendRanges shiftSendRanges(SendRanges ranges, int thisRank, LocalIndex numIncoming)
{
    for (int rank = thisRank + 1; rank <= ranges.numRanks(); ++rank)
    {
        ranges[rank] += numIncoming;
    }
    return ranges;
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
