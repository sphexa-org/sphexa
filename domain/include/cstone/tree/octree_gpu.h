/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Compute the internal part of a cornerstone octree on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#pragma once

#include "cstone/tree/octree.hpp"

namespace cstone
{

/*! @brief construct the internal octree part of a given octree leaf cell array on the GPU
 *
 * @tparam       KeyType     unsigned 32- or 64-bit integer
 * @param[in]    cstoneTree  GPU buffer with the SFC leaf cell keys
 * @param[inout] d           input:  pointers to pre-allocated GPU buffers for octree cells
 *                           output: fully linked octree
 *
 * This does not allocate memory on the GPU, (except thrust temp buffers for scans and sorting)
 */
template<class KeyType>
extern void buildOctreeGpu(const KeyType* cstoneTree, OctreeView<KeyType> d);

//! @brief same as above, but using existing buffers to avoid temporary memory allocation
template<class KeyType>
extern void buildOctreeGpu(const KeyType* cstoneTree,
                           OctreeView<KeyType> d,
                           std::span<KeyType> keyBuf,
                           std::span<TreeNodeIndex> valueBuf,
                           std::span<char> cubTmp);

//! @brief Upsweep by summing up child nodes, e.g. to compute particle node counts
void upsweepSumGpu(int numLvl, const TreeNodeIndex* lvlRange, const TreeNodeIndex* childOffsets, LocalIndex* counts);

/*!  @brief locate all nodes between k1 and k2 in nodeKeys and store indices
 * @param[in]  k1        cornerstone leaf sequence start
 * @param[in]  k2        cornerstone leaf sequence end
 * @param[in]  nodeKeys  full octree with WS-prefix-bit SFC keys per node
 * @param[in]  lvlRange  level ranges of @p nodeKeys
 * @param[out] indices   node index locations to store
 */
template<class KeyType>
extern void locateNodesGpu(const KeyType* k1,
                           const KeyType* k2,
                           const KeyType* nodeKeys,
                           const TreeNodeIndex* lvlRange,
                           TreeNodeIndex* indices);

/*!  @brief locate all nodes between k1 and k2 in nodeKeys and store indices
 * @param[in]  queryKeys  SFC keys to look up in @nodeKeys, in WS-prefix-bit format
 * @param[in]  map        index list of
 * @param[in]  n          size of @p map
 * @param[in]  nodeKeys   full octree with WS-prefix-bit SFC keys per node
 * @param[in]  lvlRange   level ranges of @p nodeKeys
 * @param[out] indices    node index locations to store
 */
template<class KeyType>
extern void locateNodesGpu(const KeyType* queryKeys,
                           const TreeNodeIndex* map,
                           size_t n,
                           const KeyType* nodeKeys,
                           const TreeNodeIndex* lvlRange,
                           TreeNodeIndex* indices);

} // namespace cstone
