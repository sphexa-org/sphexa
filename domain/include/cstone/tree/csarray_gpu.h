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

#include <span>

#include "cstone/execution.hpp"
#include "csarray.hpp"

namespace cstone
{

/*! @brief count number of particles in each octree node
 *
 * @tparam KeyType               32- or 64-bit unsigned integer type
 * @param[in]  tree              octree nodes given as Morton codes of length @a numNodes+1
 *                               needs to satisfy the octree invariants
 * @param[out] counts            output particle counts per node, length = @a numNodes
 * @param[in]  numNodes          number of nodes in tree
 * @param[in]  keys              sorted particle SFC keys
 * @param[in]  maxCount          maximum particle count per node to store, this is used
 *                               to prevent overflow in MPI_Allreduce
 * @param[in]  useCountsAsGuess  if true, use existing @p counts as starting point for the search
 * @param[in]  exec              execution policy
 */
template<class KeyType>
extern void computeNodeCountsGpu(execution::Gpu exec,
                                 const KeyType* tree,
                                 unsigned* counts,
                                 TreeNodeIndex numNodes,
                                 std::span<const KeyType> keys,
                                 unsigned maxCount,
                                 bool useCountsAsGuess);

/*! @brief split or fuse octree nodes based on node counts relative to bucketSize
 *
 * @tparam KeyType         32- or 64-bit unsigned integer type
 * @param[in] tree         vector of octree nodes in cornerstone format, length = @p numNodes + 1
 * @param[in] numNodes     number of nodes in @p tree
 * @param[in] counts       input particle counts per node, length = @p numNodes
 * @param[in] bucketSize   maximum particle count per (leaf) node
 * @param[out] nodeOps     node transformation codes, length = @p numNodes + 1
 * @param[in] exec         execution policy
 * @return                 number of nodes in the future rebalanced tree
 */
template<class KeyType>
extern TreeNodeIndex computeNodeOpsGpu(execution::Gpu exec,
                                       const KeyType* tree,
                                       TreeNodeIndex numNodes,
                                       const unsigned* counts,
                                       unsigned bucketSize,
                                       TreeNodeIndex* nodeOps);

template<class KeyType>
extern bool rebalanceTreeGpu(execution::Gpu exec,
                             const KeyType* tree,
                             TreeNodeIndex numNodes,
                             TreeNodeIndex newNumNodes,
                             const TreeNodeIndex* nodeOps,
                             KeyType* newTree);

template<class KeyType>
extern void countSfcGapsGpu(execution::Gpu exec, const KeyType* tree, TreeNodeIndex numNodes, TreeNodeIndex* nodeOps);

template<class KeyType>
extern void fillSfcGapsGpu(
    execution::Gpu exec, const KeyType* tree, TreeNodeIndex numNodes, const TreeNodeIndex* nodeOps, KeyType* newTree);

} // namespace cstone
