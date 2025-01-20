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
