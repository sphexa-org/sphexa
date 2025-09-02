/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
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

#include <iostream>
#include <vector>

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/domain/domaindecomp.hpp"
#include "cstone/primitives/primitives_acc.hpp"
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
invertRanges(TreeNodeIndex first, std::span<const IndexPair<TreeNodeIndex>> ranges, TreeNodeIndex last)
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
inline std::vector<TreeNodeIndex> enumerateRanges(std::span<const IndexPair<TreeNodeIndex>> ranges)
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
 * @param flags          select flags, length N+1
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
std::vector<IntegralType> extractMarkedElements(std::span<const IntegralType> source,
                                                std::span<const LocalIndex> flags,
                                                TreeNodeIndex firstReqIdx,
                                                TreeNodeIndex secondReqIdx)
{
    std::vector<IntegralType> requestKeys;

    while (firstReqIdx != secondReqIdx)
    {
        // advance to first halo (or to secondReqIdx)
        while (firstReqIdx < secondReqIdx && flags[firstReqIdx + 1] == flags[firstReqIdx])
        {
            firstReqIdx++;
        }

        // add one request key range
        if (firstReqIdx != secondReqIdx)
        {
            requestKeys.push_back(source[firstReqIdx]);
            // advance until not a halo or end of range
            while (firstReqIdx < secondReqIdx && flags[firstReqIdx + 1] > flags[firstReqIdx])
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
 * @param[in]  focusLeafCounts   node counts of the focus leaves, size numLeafNodes
 * @param[in]  flags             flag for each node, with a non-zero value if present as halo node, size numNodes
 * @param[in]  idx               first and last focus leaf idx of the assigned nodes on the executing rank
 * @param[out] layout            size numLeafNodes + 1. The first element is zero, the last element is
 *                               equal to the sum of all present (assigned+halo) node counts.
 */
template<bool useGpu>
void computeNodeLayout(std::span<const unsigned> focusLeafCounts,
                       std::span<const uint8_t> flags,
                       std::span<const TreeNodeIndex> leafToInternal,
                       TreeIndexPair idx,
                       std::span<LocalIndex> layout)
{
    if constexpr (useGpu)
    {
        memcpyD2D(focusLeafCounts.data() + idx.start(), idx.count(), layout.data() + idx.start());

        gatherGpu(leafToInternal.data(), idx.start(), flags.data(), layout.data());
        selectCopyGpu(focusLeafCounts.data(), idx.start(), layout.data(), layout.data());

        gatherGpu(leafToInternal.data() + idx.end(), leafToInternal.size() - idx.end(), flags.data(),
                  layout.data() + idx.end());
        selectCopyGpu(focusLeafCounts.data() + idx.end(), focusLeafCounts.size() - idx.end(), layout.data() + idx.end(),
                      layout.data() + idx.end());

        exclusiveScanGpu(layout.data(), layout.data() + layout.size(), layout.data(), LocalIndex{0});
    }
    else
    {
#pragma omp parallel for
        for (TreeNodeIndex i = 0; i < TreeNodeIndex(focusLeafCounts.size()); ++i)
        {
            bool haveParticles = (idx.start() <= i && i < idx.end()) || flags[leafToInternal[i]];
            layout[i]          = -int(haveParticles) & focusLeafCounts[i];
        }
        std::exclusive_scan(layout.begin(), layout.end(), layout.begin(), LocalIndex{0});
    }
}

//! @brief check halo discovery for sanity
template<class KeyType>
int checkLayout(int myRank,
                std::span<const TreeIndexPair> focusAssignment,
                std::span<const LocalIndex> layout,
                std::span<const KeyType> ftree)
{
    TreeNodeIndex firstNode = focusAssignment[myRank].start();
    TreeNodeIndex lastNode  = focusAssignment[myRank].end();

    std::array<TreeNodeIndex, 2> checkRanges[2] = {{0, firstNode}, {lastNode, TreeNodeIndex(nNodes(ftree))}};

    int ret = 0;
    for (int range = 0; range < 2; ++range)
    {
#pragma omp parallel for
        for (TreeNodeIndex i = checkRanges[range][0]; i < checkRanges[range][1]; ++i)
        {
            if (layout[i + 1] > layout[i])
            {
                bool peerFound = false;
                for (auto peerRange : focusAssignment)
                {
                    if (peerRange.start() <= i && i < peerRange.end()) { peerFound = true; }
                }
                if (!peerFound)
                {
                    std::cout << "Assignment rank " << myRank << " " << std::oct << ftree[firstNode] << " - "
                              << ftree[lastNode] << std::dec << std::endl;
                    std::cout << "Failed node " << i << " " << std::oct << ftree[i] << " - " << ftree[i + 1] << std::dec
                              << std::endl;
                    ret = 1;
                }
            }
        }
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
template<class... Arrays1, class... Arrays2>
void gatherArrays(std::span<const LocalIndex> ordering,
                  LocalIndex outputOffset,
                  std::tuple<Arrays1&...> arrays,
                  std::tuple<Arrays2&...> scratchBuffers)
{
    auto reorderArray = [ordering, outputOffset, &scratchBuffers](auto& array)
    {
        using VectorRef  = decltype(array);
        using VectorType = std::decay_t<VectorRef>;
        if constexpr (util::Contains<VectorRef, std::tuple<Arrays2&...>>{})
        {
            auto& swapSpace = util::pickType<decltype(array)>(scratchBuffers);
            assert(swapSpace.size() == array.size());
            gatherAcc<IsDeviceVector<VectorType>{}>(ordering, rawPtr(array), rawPtr(swapSpace) + outputOffset);
            swap(swapSpace, array);
        }
        else
        {
            constexpr int i = util::FindIndex<VectorRef, std::tuple<Arrays2&...>, SmallerElementSize>{};
            static_assert(i < sizeof...(Arrays2));
            assert(std::get<i>(scratchBuffers).size() == array.size());

            auto* scratch = reinterpret_cast<typename VectorType::value_type*>(rawPtr(std::get<i>(scratchBuffers)));
            gatherAcc<IsDeviceVector<VectorType>{}>(ordering, rawPtr(array), scratch);
            copy_n<IsDeviceVector<VectorType>{}>(scratch, ordering.size(), rawPtr(array) + outputOffset);
        }
    };

    util::for_each_tuple(reorderArray, arrays);
}

} // namespace cstone
