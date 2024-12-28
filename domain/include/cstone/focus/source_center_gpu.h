/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Compute cell mass centers for use in focus tree
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/focus/source_center.hpp"
#include "cstone/tree/definitions.h"

namespace cstone
{

/*! @brief compute mass centers of leaf cells
 *
 * @param[in]  x                particle x coordinates
 * @param[in]  y                particle y coordinates
 * @param[in]  z                particle z coordinates
 * @param[in]  m                particle masses
 * @param[in]  leafToInternal   maps a leaf node index to an internal layout node index
 * @param[in]  numLeaves        number of leaf nodes
 * @param[in]  layout           particle location of each node, length @a numLeaves + 1
 * @param[out] centers          output mass centers, in internal node layout, length >= max(leafToInternal)
 */
template<class Tc, class Tm, class Tf>
extern void computeLeafSourceCenterGpu(const Tc* x,
                                       const Tc* y,
                                       const Tc* z,
                                       const Tm* m,
                                       const TreeNodeIndex* leafToInternal,
                                       TreeNodeIndex numLeaves,
                                       const LocalIndex* layout,
                                       Vec4<Tf>* centers);

/*! @brief compute center of gravity for internal nodes with an upsweep
 *
 * @tparam T                   float or double
 * @param[in]    numLevels     max tree depth
 * @param[in]    levelRange    first node index per tree level
 * @param[in]    childOffsets  indices of first child node of each node
 * @param[inout] centers       center of mass coordinates with leaf node centers set
 */
template<class T>
extern void upsweepCentersGpu(int numLevels,
                              const TreeNodeIndex* levelRange,
                              const TreeNodeIndex* childOffsets,
                              SourceCenterType<T>* centers);

//! @brief compute geometric node center and sizes based on node SFC keys
template<class KeyType, class T>
extern void computeGeoCentersGpu(
    const KeyType* prefixes, TreeNodeIndex numNodes, Vec3<T>* centers, Vec3<T>* sizes, const Box<T>& box);

//! @brief set @p centers to geometric node centers with Mac radius l * invTheta
template<class KeyType, class T>
extern void geoMacSpheresGpu(
    const KeyType* prefixes, TreeNodeIndex numNodes, SourceCenterType<T>* centers, float invTheta, const Box<T>& box);

template<class KeyType, class T>
extern void
setMacGpu(const KeyType* prefixes, TreeNodeIndex numNodes, Vec4<T>* macSpheres, float invTheta, const Box<T>& box);

template<class T>
extern void moveCenters(const Vec3<T>* src, TreeNodeIndex numNodes, Vec4<T>* dest);

} // namespace cstone
