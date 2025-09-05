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

#include "cstone/focus/source_center.hpp"
#include "cstone/traversal/boxoverlap.hpp"
#include "cstone/traversal/traversal.hpp"

namespace cstone
{

template<class KeyType, class T>
HOST_DEVICE_FUN void findCollisions(const KeyType* nodePrefixes,
                                    const TreeNodeIndex* childOffsets,
                                    const TreeNodeIndex* parents,
                                    const Vec3<T>* nodeCenters,
                                    const Vec3<T>* nodeSizes,
                                    const Vec3<T> targetCenter,
                                    const Vec3<T> targetSize,
                                    const Box<T>& box,
                                    KeyType excludeStart,
                                    KeyType excludeEnd,
                                    uint8_t* flags)
{
    auto overlaps = [&](TreeNodeIndex idx)
    {
        auto [nk1, nk2] = decodePlaceholderBit2K(nodePrefixes[idx]);
        bool bOverlap   = !containedIn(nk1, nk2, excludeStart, excludeEnd) &&
                        overlap(nodeCenters[idx], nodeSizes[idx], targetCenter, targetSize, box);
        if (bOverlap) { flags[idx] = 1; }
        return bOverlap;
    };

    singleTraversal(childOffsets, parents, overlaps, [](TreeNodeIndex) {});
}


template<class KeyType, class F>
HOST_DEVICE_FUN void findCollisions(const KeyType* nodePrefixes,
                                    const TreeNodeIndex* childOffsets,
                                    const TreeNodeIndex* parents,
                                    F&& endpointAction,
                                    const IBox& target,
                                    KeyType excludeStart,
                                    KeyType excludeEnd,
                                    unsigned bx,
                                    unsigned by,
                                    unsigned bz)
{
    auto overlaps = [excludeStart, excludeEnd, nodePrefixes, &target, &bx, &by, &bz](TreeNodeIndex idx)
    {
        KeyType nodeKey = decodePlaceholderBit(nodePrefixes[idx]);
        int level       = decodePrefixLength(nodePrefixes[idx]) / 3;
        IBox sourceBox  = sfcIBox(sfcMixDKey(nodeKey), maxTreeLevel<KeyType>{} - level, bx, by, bz);
        if (sourceBox.xmin() == sourceBox.xmax() && sourceBox.ymin() == sourceBox.ymax() &&
            sourceBox.zmin() == sourceBox.zmax())
        {
            // if the cell is empty, we return no overlap
            return false;
        }
        // std::cout << "[findCollisions] idx: " << idx << " sourceBox: " << sourceBox.xmin() << ", " << sourceBox.xmax() << ", "
        //           << sourceBox.ymin() << ", " << sourceBox.ymax() << ", " << sourceBox.zmin() << ", " << sourceBox.zmax() << std::endl;
        // std::cout << "[findCollisions] idx: " << idx << " target:    " << target.xmin() << ", " << target.xmax() << ", "
        //           << target.ymin() << ", " << target.ymax() << ", " << target.zmin() << ", " << target.zmax() << std::endl;
        // std::cout << "[findCollisions] idx: " << idx << " excludeStart: " << std::oct << excludeStart
        //       << ", excludeEnd: " << excludeEnd << std::dec << std::endl;
        // std::cout << "[findCollisions] idx: " << idx << " nodeKey: " << std::oct << nodeKey << std::dec << ", level: " << level << std::endl;
        // std::cout << "[findCollisions] idx: " << idx << " nodeKey + nodeRange<KeyType>(level): " << std::oct << nodeKey + nodeRange<KeyType>(level) << std::dec << std::endl;
        // std::cout << "[findCollisions] idx: " << idx << " containedIn: " << containedIn(nodeKey, nodeKey + nodeRange<KeyType>(level), excludeStart, excludeEnd) << std::endl;
        // std::cout << "[findCollisions] idx: " << idx << " overlap: " << overlap<KeyType>(sourceBox, target) << std::endl;
        return !containedIn(nodeKey, nodeKey + nodeRange<KeyType>(level), excludeStart, excludeEnd) &&
               overlap<KeyType>(sourceBox, target);
    };

    singleTraversal(childOffsets, parents, overlaps, endpointAction);
}

/*! @brief mark halo nodes with flags
 *
 * @tparam KeyType               32- or 64-bit unsigned integer
 * @tparam Tc                    float or double
 * @param[in]  prefixes          node keys in placeholder-bit format of fully linked octree
 * @param[in]  childOffsets      first child node index of each node
 * @param[in]  leaves            cornerstone array of tree leaves
 * @param[in]  searchCenters     effective halo search box center per octree (leaf) node
 * @param[in]  searchSizes       effective halo search box size per octree (leaf) node
 * @param[in]  box               coordinate bounding box
 * @param[in]  firstNode         first leaf node index to consider as local
 * @param[in]  lastNode          last leaf node index to consider as local
 * @param[out] collisionFlags    array of length octree.numTreeNodes, each node that is a halo
 *                               from the perspective of [firstNode:lastNode] will be marked
 *                               with a non-zero value.
 *                               Note: does NOT reset non-colliding indices to 0, so @p collisionFlags
 *                               should be zero-initialized prior to calling this function.
 */
template<class KeyType, class Tc>
void findHalos(const KeyType* prefixes,
               const TreeNodeIndex* childOffsets,
               const TreeNodeIndex* parents,
               const Vec3<Tc>* nodeCenters,
               const Vec3<Tc>* nodeSizes,
               const KeyType* leaves,
               const Vec3<Tc>* searchCenters,
               const Vec3<Tc>* searchSizes,
               const Box<Tc>& box,
               TreeNodeIndex firstNode,
               TreeNodeIndex lastNode,
               uint8_t* collisionFlags)
{
    KeyType lowestKey  = leaves[firstNode];
    KeyType highestKey = leaves[lastNode];

    const auto mixDBits = getBoxMixDimensionBits<Tc, KeyType, Box<Tc>>(box);
    const bool use_mixD = mixDBits.bx != maxTreeLevel<KeyType>{} ||
                    mixDBits.by != maxTreeLevel<KeyType>{} ||
                    mixDBits.bz != maxTreeLevel<KeyType>{};


#pragma omp parallel for
    for (TreeNodeIndex leafIdx = firstNode; leafIdx < lastNode; ++leafIdx)
    {
        // if the halo box is fully inside the assigned SFC range, we skip collision detection
        if (use_mixD && containedIn(lowestKey, highestKey, searchCenters[leafIdx], searchSizes[leafIdx], box, mixDBits.bx, mixDBits.by, mixDBits.bz)) { continue; }
        if (!use_mixD && containedIn(lowestKey, highestKey, searchCenters[leafIdx], searchSizes[leafIdx], box)) { continue; }

        findCollisions(prefixes, childOffsets, parents, nodeCenters, nodeSizes, searchCenters[leafIdx],
                       searchSizes[leafIdx], box, lowestKey, highestKey, collisionFlags);
    }
}

} // namespace cstone
