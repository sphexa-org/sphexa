/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Zurich, 2021 University of Basel
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Utility functions for determining the layout of particle buffers on a given rank
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * Each rank will be assigned a part of the SFC, equating to one or multiple ranges of
 * node indices of the global cornerstone octree. In a addition to the assigned nodes,
 * each rank must also store particle data for those nodes in the global octree which are
 * halos of the assigned nodes. Both types of nodes present on the rank are stored in the same
 * particle array (x,y,z,h,...) according to increasing node index, which is the same
 * as increasing Morton code.
 *
 * Given
 *  - the global cornerstone tree
 *  - its assignment to ranks
 *  - lists of in/outgoing halo nodes (global indices) per rank,
 * the utility functions in this file determine the position and size of each node (halo or assigned node)
 * in the particle buffers. The resulting layout is valid for all particle buffers, such as x,y,z,h,d,p,...
 *
 * Note:
 * If a node of the global cornerstone octree has index i, this means its Morton code range is tree[i] - tree[i+1]
 */

#pragma once

#include <vector>

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/domain/domaindecomp.hpp"
#include "cstone/util/tuple_util.hpp"
#include "cstone/util/type_list.hpp"

namespace cstone
{

/*! @brief calculates the complementary range of the input ranges
 *
 * Input:  │      ------    -----   --     ----     --  │
 * Output: -------      ----     ---  -----    -----  ---
 *         ^                                            ^
 *         │                                            │
 * @param first                                         │
 * @param ranges   size >= 1, must be sorted            │
 * @param last    ──────────────────────────────────────/
 * @return the output ranges that cover everything within [first:last]
 *         that the input ranges did not cover
 */
inline std::vector<IndexPair<TreeNodeIndex>>
invertRanges(TreeNodeIndex first, gsl::span<const IndexPair<TreeNodeIndex>> ranges, TreeNodeIndex last)
{
    std::vector<IndexPair<TreeNodeIndex>> invertedRanges;

    TreeNodeIndex currentIndex = first;
    for (auto range : ranges)
    {
        if (range.start() == range.end()) { continue; }

        assert(currentIndex <= range.start() && "non-empty ranges must be sorted\n");
        if (currentIndex < range.start()) { invertedRanges.emplace_back(currentIndex, range.start()); }
        currentIndex = range.end();
    }
    if (currentIndex < last) { invertedRanges.emplace_back(currentIndex, last); }

    return invertedRanges;
}

//! @brief enumerate all input ranges with std::iota and pack them together in a single vector
inline std::vector<TreeNodeIndex> enumerateRanges(gsl::span<const IndexPair<TreeNodeIndex>> ranges)
{
    std::vector<TreeNodeIndex> rangeCounts(ranges.size() + 1);
    std::transform(ranges.begin(), ranges.end(), rangeCounts.begin(), [](auto pair) { return pair.count(); });
    std::exclusive_scan(rangeCounts.begin(), rangeCounts.end(), rangeCounts.begin(), 0);

    std::vector<TreeNodeIndex> ret(rangeCounts.back());
    for (size_t i = 0; i < ranges.size(); ++i)
    {
        std::iota(ret.begin() + rangeCounts[i], ret.begin() + rangeCounts[i] + ranges[i].count(), ranges[i].start());
    }
    return ret;
}

/*! @brief extract ranges of marked indices from a source array
 *
 * @tparam IntegralType  an integer type
 * @param source         array with quantities to extract, length N+1
 * @param flags          0 or 1 flags for index, length N
 * @param firstReqIdx    first index, permissible range: [0:N]
 * @param secondReqIdx   second index, permissible range: [0:N+1]
 * @return               vector (of pairs) of elements of @p source that span all
 *                       elements [firstReqIdx:secondReqIdx] of @p source that are
 *                       marked by @p flags
 *
 * Even indices mark the start of a range, uneven indices mark the end of the previous
 * range start. If two ranges are consecutive, they are fused into a single range.
 *
 * This is used to extract
 *  - SFC keys of cornerstone octree leaf nodes flagged as halos
 *  - Particle offsets from buffer layouts
 */
template<class IntegralType>
std::vector<IntegralType> extractMarkedElements(gsl::span<const IntegralType> source,
                                                gsl::span<const int> flags,
                                                TreeNodeIndex firstReqIdx,
                                                TreeNodeIndex secondReqIdx)
{
    std::vector<IntegralType> requestKeys;

    while (firstReqIdx != secondReqIdx)
    {
        // advance to first halo (or to secondReqIdx)
        while (firstReqIdx < secondReqIdx && flags[firstReqIdx] == 0)
        {
            firstReqIdx++;
        }

        // add one request key range
        if (firstReqIdx != secondReqIdx)
        {
            requestKeys.push_back(source[firstReqIdx]);
            // advance until not a halo or end of range
            while (firstReqIdx < secondReqIdx && flags[firstReqIdx] == 1)
            {
                firstReqIdx++;
            }
            requestKeys.push_back(source[firstReqIdx]);
        }
    }

    return requestKeys;
}

/*! @brief calculate the location (offset) of each focus tree leaf node in the particle arrays
 *
 * @param[in]  focusLeafCounts   node counts of the focus leaves, size N
 * @param[in]  haloFlags         flag for each node, with a non-zero value if present as halo node, size N
 * @param[in]  firstAssignedIdx  first focus leaf idx to treat as part of the assigned nodes on the executing rank
 * @param[in]  lastAssignedIdx   last focus leaf idx to treat as part of the assigned nodes on the executing rank
 * @param[out] layout            length N+1. The first element is zero, the last element is
 *                               equal to the sum of all all present (assigned+halo) node counts.
 */
inline void computeNodeLayout(gsl::span<const unsigned> focusLeafCounts,
                              gsl::span<const int> haloFlags,
                              TreeNodeIndex firstAssignedIdx,
                              TreeNodeIndex lastAssignedIdx,
                              gsl::span<LocalIndex> layout)
{
#pragma omp parallel for
    for (TreeNodeIndex i = 0; i < TreeNodeIndex(focusLeafCounts.size()); ++i)
    {
        bool haveParticles = (firstAssignedIdx <= i && i < lastAssignedIdx) || haloFlags[i];
        layout[i]          = -int(haveParticles) & focusLeafCounts[i];
    }

    exclusiveScan(layout.data(), layout.size());
}

/*! @brief computes a list which local array ranges are going to be filled with halo particles
 *
 * @param layout       prefix sum of leaf counts of locally present nodes (see computeNodeLayout)
 *                     length N+1
 * @param haloFlags    0 or 1 for each leaf, length N
 * @param assignment   assignment of leaf nodes to peer ranks
 * @param peerRanks    list of peer ranks
 * @return             list of array index ranges for the receiving part in exchangeHalos
 */
inline auto computeHaloRecvList(gsl::span<const LocalIndex> layout,
                                gsl::span<const int> haloFlags,
                                gsl::span<const TreeIndexPair> assignment,
                                gsl::span<const int> peerRanks)
{
    RecvList ret(assignment.size());

    for (int peer : peerRanks)
    {
        auto pFlags           = haloFlags.subspan(assignment[peer].start(), assignment[peer].count());
        TreeNodeIndex firstNz = std::distance(haloFlags.begin(), std::find(pFlags.begin(), pFlags.end(), 1));
        TreeNodeIndex lastNz  = std::distance(haloFlags.begin(), std::find(pFlags.rbegin(), pFlags.rend(), 1).base());
        ret[peer]             = {layout[firstNz], layout[lastNz]};
    }

    return ret;
}

//! @brief Compare value_type size of container T to the value_type size of the N-th container in Tuple
template<int N, class T, class Tuple>
struct SmallerElementSize
    : std::bool_constant<sizeof(typename std::decay_t<T>::value_type) <=
                         sizeof(typename std::decay_t<std::tuple_element_t<N, Tuple>>::value_type)>
{
};

//! @brief reorder with state-less function object
template<class Gather, class... Arrays1, class... Arrays2>
void gatherArrays(Gather&& gatherFunc,
                  const LocalIndex* ordering,
                  LocalIndex numElements,
                  LocalIndex inputOffset,
                  LocalIndex outputOffset,
                  std::tuple<Arrays1&...> arrays,
                  std::tuple<Arrays2&...> scratchBuffers)
{
    auto reorderArray = [ordering, numElements, inputOffset, outputOffset, &gatherFunc, &scratchBuffers](auto& array)
    {
        using VectorRef = decltype(array);
        if constexpr (util::Contains<VectorRef, std::tuple<Arrays2&...>>{})
        {
            auto& swapSpace = util::pickType<decltype(array)>(scratchBuffers);
            assert(swapSpace.size() == array.size());
            gatherFunc(ordering, numElements, rawPtr(array) + inputOffset, rawPtr(swapSpace) + outputOffset);
            swap(swapSpace, array);
        }
        else
        {
            constexpr int i = util::FindIndex<VectorRef, std::tuple<Arrays2&...>, SmallerElementSize>{};
            static_assert(i < sizeof...(Arrays2));
            assert(std::get<i>(scratchBuffers).size() == array.size());

            auto* scratchSpace =
                reinterpret_cast<typename std::decay_t<VectorRef>::value_type*>(rawPtr(std::get<i>(scratchBuffers)));
            gatherFunc(ordering, numElements, rawPtr(array) + inputOffset, scratchSpace);
            if constexpr (IsDeviceVector<std::decay_t<VectorRef>>{})
            {
                memcpyD2D(scratchSpace, numElements, rawPtr(array) + outputOffset);
            }
            else { omp_copy(scratchSpace, scratchSpace + numElements, rawPtr(array) + outputOffset); }
        }
    };

    util::for_each_tuple(reorderArray, arrays);
}

} // namespace cstone
