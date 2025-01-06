/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Collision detection for halo discovery using octree traversal
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/traversal/boxoverlap.hpp"
#include "cstone/traversal/traversal.hpp"

namespace cstone
{

template<class KeyType, class F>
HOST_DEVICE_FUN void findCollisions(const KeyType* nodePrefixes,
                                    const TreeNodeIndex* childOffsets,
                                    F&& endpointAction,
                                    const IBox& target,
                                    KeyType excludeStart,
                                    KeyType excludeEnd)
{
    auto overlaps = [excludeStart, excludeEnd, nodePrefixes, &target](TreeNodeIndex idx)
    {
        KeyType nodeKey = decodePlaceholderBit(nodePrefixes[idx]);
        int level       = decodePrefixLength(nodePrefixes[idx]) / 3;
        IBox sourceBox  = sfcIBox(sfcKey(nodeKey), level);
        return !containedIn(nodeKey, nodeKey + nodeRange<KeyType>(level), excludeStart, excludeEnd) &&
               overlap<KeyType>(sourceBox, target);
    };

    singleTraversal(childOffsets, overlaps, endpointAction);
}

/*! @brief mark halo nodes with flags
 *
 * @tparam KeyType               32- or 64-bit unsigned integer
 * @tparam RadiusType            float or double, float is sufficient for 64-bit codes or less
 * @tparam CoordinateType        float or double
 * @param[in]  prefixes          node keys in placeholder-bit format of fully linked octree
 * @param[in]  childOffsets      first child node index of each node
 * @param[in]  internalToLeaf    conversion of fully linked indices to cstone indices
 * @param[in]  leaves            cornerstone array of tree leaves
 * @param[in]  interactionRadii  effective halo search radii per octree (leaf) node
 * @param[in]  box               coordinate bounding box
 * @param[in]  firstNode         first leaf node index to consider as local
 * @param[in]  lastNode          last leaf node index to consider as local
 * @param[out] collisionFlags    array of length octree.numLeafNodes, each node that is a halo
 *                               from the perspective of [firstNode:lastNode] will be marked
 *                               with a non-zero value.
 *                               Note: does NOT reset non-colliding indices to 0, so @p collisionFlags
 *                               should be zero-initialized prior to calling this function.
 */
template<class KeyType, class RadiusType, class CoordinateType>
void findHalos(const KeyType* prefixes,
               const TreeNodeIndex* childOffsets,
               const TreeNodeIndex* internalToLeaf,
               const KeyType* leaves,
               const RadiusType* interactionRadii,
               const Box<CoordinateType>& box,
               TreeNodeIndex firstNode,
               TreeNodeIndex lastNode,
               uint8_t* collisionFlags)
{
    KeyType lowestCode  = leaves[firstNode];
    KeyType highestCode = leaves[lastNode];

    auto markCollisions = [collisionFlags, internalToLeaf](TreeNodeIndex i) { collisionFlags[internalToLeaf[i]] = 1; };

#pragma omp parallel for
    for (TreeNodeIndex nodeIdx = firstNode; nodeIdx < lastNode; ++nodeIdx)
    {
        RadiusType radius = interactionRadii[nodeIdx];
        IBox haloBox      = makeHaloBox<KeyType>(leaves[nodeIdx], leaves[nodeIdx + 1], radius, box);

        // if the halo box is fully inside the assigned SFC range, we skip collision detection
        if (containedIn(lowestCode, highestCode, haloBox)) { continue; }

        findCollisions(prefixes, childOffsets, markCollisions, haloBox, lowestCode, highestCode);
    }
}

} // namespace cstone