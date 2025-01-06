/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  GPU driver for halo discovery using traversal of an octree
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/traversal/collisions.hpp"

namespace cstone
{

/*! @brief mark halo nodes with flags
 *
 * @tparam KeyType               32- or 64-bit unsigned integer
 * @tparam RadiusType            float or double, float is sufficient for 64-bit codes or less
 * @tparam T                     float or double
 * @param[in]  prefixes          Warren-Salmon node keys of the octree, length = numTreeNodes
 * @param[in]  childOffsets      child offsets array, length = numTreeNodes
 * @param[in]  internalToLeaf    map leaf node indices of fully linked format to cornerstone order
 * @param[in]  leaves            cstone array of leaf node keys
 * @param[in]  interactionRadii  effective halo search radii per octree (leaf) node
 * @param[in]  box               coordinate bounding box
 * @param[in]  firstNode         first cstone leaf node index to consider as local
 * @param[in]  lastNode          last cstone leaf node index to consider as local
 * @param[out] collisionFlags    array of length numLeafNodes, each node that is a halo
 *                               from the perspective of [firstNode:lastNode] will be marked
 *                               with a non-zero value.
 *                               Note: does NOT reset non-colliding indices to 0, so @p collisionFlags
 *                               should be zero-initialized prior to calling this function.
 */
template<class KeyType, class RadiusType, class T>
extern void findHalosGpu(const KeyType* prefixes,
                         const TreeNodeIndex* childOffsets,
                         const TreeNodeIndex* internalToLeaf,
                         const KeyType* leaves,
                         const RadiusType* interactionRadii,
                         const Box<T>& box,
                         TreeNodeIndex firstNode,
                         TreeNodeIndex lastNode,
                         int* collisionFlags);

template<class T, class KeyType>
extern void markMacsGpu(const KeyType* prefixes,
                        const TreeNodeIndex* childOffsets,
                        const Vec4<T>* centers,
                        const Box<T>& box,
                        const KeyType* focusNodes,
                        TreeNodeIndex numFocusNodes,
                        bool limitSource,
                        uint8_t* markings);

} // namespace cstone
