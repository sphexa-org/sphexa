/*
 * MIT License
 *
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
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
 * @brief Barnes-Hut breadth-first warp-aware tree traversal inspired by the original Bonsai implementation
 *
 * @author Rio Yokota <rioyokota@gsic.titech.ac.jp>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/traversal/groups.hpp"
#include "cstone/util/array.hpp"

namespace ryoanji
{

/*! @brief Compute approximate body accelerations with Barnes-Hut
 *
 * @param[in]    grp            groups of target particles to compute accelerations for
 * @param[in]    initNodeIdx    traversal will be started with all children of the parent of @p initNodeIdx
 * @param[in]    x,y,z,m,h      bodies, in SFC order and as referenced by sourceCells
 * @param[in]    childOffsets   location (index in [0:numTreeNodes]) of first child of each cell, 0 indicates a leaf
 * @param[in]    internalToLeaf for each cell in [0:numTreeNodes], stores the leaf cell (cstone) index in [0:numLeaves]
 *                              if the cell is not a leaf, the value is negative
 * @param[in]    layout         for each leaf cell in [0:numLeaves], stores the index of the first body in the cell
 * @param[in]    sourceCenter   x,y,z center and square MAC radius of each cell in [0:numTreeNodes]
 * @param[in]    Multipole      cell multipoles, on device
 * @param[in]    G              gravitational constant
 * @param[in]    numShells      number of periodic replicas in each dimension to include
 * @param[in]    boxL           length of coordinate bounding box in each dimension
 * @param[inout] p              output body potential to add to if not nullptr
 * @param[inout] ax, ay, az     output body acceleration to add to
 * @param[-]     gmPool         temporary storage for the cell traversal stack, uninitialized
 *                              each active warp needs space for TravConfig::memPerWarp int32,
 *                              so the total size is TravConfig::memPerWarp * numWarpsPerBlock * numBlocks
 * @return                      total potential
 */
template<class Tc, class Th, class Tm, class Ta, class Tf, class MType>
extern double traverse(cstone::GroupView grp, const int initNodeIdx, const Tc* x, const Tc* y, const Tc* z, const Tm* m,
                       const Th* h, const TreeNodeIndex* childOffsets, const TreeNodeIndex* internalToLeaf,
                       const LocalIndex* layout, const Vec4<Tf>* sourceCenter, const MType* Multipoles, Tc G,
                       int numShells, Vec3<Tc> boxL, Ta* p, Ta* ax, Ta* ay, Ta* az, int* gmPool);

//! @brief maximum number of particles per target group that the "traverse" function can handle
int bhMaxTargetSize();

//! @brief compute required traversal stack size (gmPool) for the "traverse" function
LocalIndex stackSize(LocalIndex numGroups);

//! @brief return traversal statistics from last call to "traverse": sumP2P, maxP2P, sumM2P, maxM2P, maxStack
util::array<uint64_t, 5> readBhStats();

} // namespace ryoanji
