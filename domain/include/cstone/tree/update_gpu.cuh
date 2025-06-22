/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Generation of local and global octrees in cornerstone format on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * See octree.hpp for a description of the cornerstone format.
 */

#pragma once

#include <limits>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/tree/csarray_gpu.h"
#include "csarray.hpp"

namespace cstone
{

/*! @brief update the octree with a single rebalance/count step
 *
 * @tparam KeyType           32- or 64-bit unsigned integer for morton code
 * @param[in]    keys        local particle SFC keys
 * @param[in]    bucketSize  maximum number of particles per node
 * @param[inout] tree        the octree leaf nodes (cornerstone format)
 * @param[inout] counts      the octree leaf node particle count
 * @param[-]     tmpTree     temporary array, will be resized as needed
 * @param[-]     workArray   temporary array, will be resized as needed
 * @param[in]    maxCount    if actual node counts are higher, they will be capped to @p maxCount
 * @return                   true if converged, false otherwise
 */
template<class KeyType, class DevKeyVec, class DevCountVec, class DevIdxVec>
bool updateOctreeGpu(std::span<const KeyType> keys,
                     unsigned bucketSize,
                     DevKeyVec& tree,
                     DevCountVec& counts,
                     DevKeyVec& tmpTree,
                     DevIdxVec& workArray,
                     unsigned maxCount = std::numeric_limits<unsigned>::max())
{
    workArray.resize(tree.size());
    TreeNodeIndex newNumNodes =
        computeNodeOpsGpu(rawPtr(tree), nNodes(tree), rawPtr(counts), bucketSize, rawPtr(workArray));

    tmpTree.resize(newNumNodes + 1);
    bool converged = rebalanceTreeGpu(rawPtr(tree), nNodes(tree), newNumNodes, rawPtr(workArray), rawPtr(tmpTree));

    swap(tree, tmpTree);
    counts.resize(nNodes(tree));

    // local node counts
    computeNodeCountsGpu(rawPtr(tree), rawPtr(counts), nNodes(tree), keys, maxCount, true);

    return converged;
}

} // namespace cstone
