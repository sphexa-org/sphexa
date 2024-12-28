/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Spatial target particle grouping
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/cuda/device_vector.h"
#include "cstone/sfc/box.hpp"
#include "cstone/traversal/groups.hpp"

namespace cstone
{

/*! @brief set up fixed-size particle groups
 * @param[in]  first      first local particle index
 * @param[in]  last       last local particle index
 * @param[in]  groupSize  number of particles per group
 * @param[out] groups     groups with fixed size @p groupSize
 */
void computeFixedGroups(LocalIndex first, LocalIndex last, unsigned groupSize, GroupData<GpuTag>& groups);

/*!* @brief Compute groups of particles with a maximum size and distance between consecutive particles limited
 *
 * @tparam groupSize                integer
 * @tparam Tc                       float or double
 * @tparam T
 * @tparam KeyType
 * @param[in]  first                index of first particle in @p x,y,z,h buffers assigned to local domain
 * @param[in]  last                 index of ilast particle in @p x,y,z,h buffers assigned to local domain
 * @param[in]  x                    x coordinates
 * @param[in]  y                    y coordinates
 * @param[in]  z                    z coordinates
 * @param[in]  h                    smoothin lengths
 * @param[in]  leaves               cornerstone leaf array, size is @numLeaves + 1
 * @param[in]  numLeaves            number of leaves in @p leaves
 * @param[in]  layout               element i stores the particle index of the first particle of leaf cell i
 * @param[in]  box                  global coordinate bounding box
 * @param[in]  groupSize            maximum number of particles per group
 * @param[in]  tolFactor            tolerance factor for maximum distance between consecutive particles in group
 * @param[-]   splitMasks           temporary scratch usage
 * @param[-]   numSplitsPerGroup    temporary scratch usage
 * @param[out] groups               particle groups stored as indices into coordinate arrays
 *
 * This function first creates groups of fixed size @param groupSize. The resulting groups will be split into smaller
 * groups until no distance between consecutive particles is bigger than @p tolFactor times edge length of the smallest
 * leaf cell of any particle in the group. Edge length is computed as the cubic root of the cell volume.
 */
template<class Tc, class T, class KeyType>
extern void computeGroupSplits(LocalIndex first,
                               LocalIndex last,
                               const Tc* x,
                               const Tc* y,
                               const Tc* z,
                               const T* h,
                               const KeyType* leaves,
                               TreeNodeIndex numLeaves,
                               const LocalIndex* layout,
                               const Box<Tc> box,
                               unsigned groupSize,
                               float tolFactor,
                               DeviceVector<LocalIndex>& numSplitsPerGroup,
                               DeviceVector<LocalIndex>& groups);

} // namespace cstone
