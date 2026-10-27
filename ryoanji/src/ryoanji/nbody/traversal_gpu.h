/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Barnes-Hut breadth-first warp-aware tree traversal inspired by the original Bonsai implementation
 *
 * @author Rio Yokota <rioyokota@gsic.titech.ac.jp>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/traversal/groups.hpp"
#include "cstone/util/array.hpp"

namespace ryoanji
{

/*! @brief Compute approximate body accelerations with Barnes-Hut
 *
 * @param[in]    grp            groups of target particles to compute accelerations for
 * @param[in]    initNodeIdx    traversal will be started with all children of the parent of @p initNodeIdx
 * @param[in]    xt,yt,zt,mt,ht target bodies, in SFC order and as referenced by grp, can be identical to sources
 * @param[in]    xs,ys,zs,ms,hs target bodies, in SFC order and as referenced by layout
 * @param[in]    childOffsets   location (index in [0:numTreeNodes]) of first child of each cell, 0 indicates a leaf
 * @param[in]    internalToLeaf for each cell in [0:numTreeNodes], stores the leaf cell (cstone) index in [0:numLeaves]
 *                              if the cell is not a leaf, the value is negative
 * @param[in]    layout         for each leaf cell in [0:numLeaves], stores the index of the first body in the cell
 * @param[in]    sourceCenter   x,y,z center and square MAC radius of each cell in [0:numTreeNodes]
 * @param[in]    Multipoles     cell multipoles, on device
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
extern double traverse(cstone::GroupView grp, int initNodeIdx, const Tc* xt, const Tc* yt, const Tc* zt, const Tm* mt,
                       const Th* ht, const Tc* xs, const Tc* ys, const Tc* zs, const Tm* ms, const Th* hs,
                       const TreeNodeIndex* childOffsets, const TreeNodeIndex* internalToLeaf, const LocalIndex* layout,
                       const Vec4<Tf>* sourceCenter, const MType* Multipoles, Tc G, int numShells, Vec3<Tc> boxL, Ta* p,
                       Ta* ax, Ta* ay, Ta* az, int* gmPool);

/*! @brief Compute approximate body accelerations with Barnes-Hut with in-register group splitting
 *
 * Identical to the group-based overload of "traverse", except that the target groups are
 * determined inside the traversal kernel: each warp claims one fixed group of
 * TravConfig::targetSize particles, evaluates the spatial compactness criterion in-register
 * with the shared cstone device function also used by the standalone split kernels, decodes its
 * subgroup boundaries and traverses the subgroups sequentially. Targets and sources are the same
 * particle set. No precomputed groups are required and the number of groups is computable on the
 * host from @p grp, so no device-to-host synchronization precedes the launch.
 *
 * @param[in]    grp            target group view, typically a trivial view (fixed-size groups)
 * @param[in]    leaves         cornerstone leaf-cell array, size is @p numLeaves + 1, on device
 * @param[in]    numLeaves      number of leaf cells in @p leaves
 * @param[in]    layout         for each leaf cell, stores the index of the first body in the cell
 * @param[in]    box            global coordinate bounding box
 * @param[in]    tolFactor      tolerance factor for the spatial compactness criterion
 * @param[in]    initNodeIdx    traversal will be started with all children of the parent of @p initNodeIdx
 * @param[in]    x,y,z,m,h      target and source bodies, in SFC order as referenced by layout, on device
 * @param[in]    childOffsets   location (index in [0:numTreeNodes]) of first child of each cell, 0 indicates a leaf
 * @param[in]    internalToLeaf for each cell in [0:numTreeNodes], stores the leaf cell (cstone) index in [0:numLeaves]
 *                              if the cell is not a leaf, the value is negative
 * @param[in]    sourceCenter   x,y,z center and square MAC radius of each cell in [0:numTreeNodes]
 * @param[in]    Multipoles     cell multipoles, on device
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
template<class Tc, class Th, class Tm, class Ta, class Tf, class MType, class KeyType>
extern double traverse(cstone::GroupView grp, const KeyType* leaves, TreeNodeIndex numLeaves,
                       const LocalIndex* layout, const cstone::Box<Tc>& box, float tolFactor, int initNodeIdx,
                       const Tc* x, const Tc* y, const Tc* z, const Tm* m, const Th* h,
                       const TreeNodeIndex* childOffsets, const TreeNodeIndex* internalToLeaf,
                       const Vec4<Tf>* sourceCenter, const MType* Multipoles, Tc G, int numShells, Vec3<Tc> boxL,
                       Ta* p, Ta* ax, Ta* ay, Ta* az, int* gmPool);

//! @brief compute required traversal stack size (gmPool) for the "traverse" function
LocalIndex stackSize(LocalIndex numGroups);

//! @brief return traversal statistics from last call to "traverse": sumP2P, maxP2P, sumM2P, maxM2P, maxStack
util::array<uint64_t, 5> readBhStats();

} // namespace ryoanji
