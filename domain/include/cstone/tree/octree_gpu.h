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

//! @brief Upsweep by summing up child nodes, e.g. to compute particle node counts
void upsweepSumGpu(int numLvl, const TreeNodeIndex* lvlRange, const TreeNodeIndex* childOffsets, LocalIndex* counts);

} // namespace cstone
