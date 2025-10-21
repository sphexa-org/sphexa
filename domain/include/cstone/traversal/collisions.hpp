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
    constexpr T epsilon = std::numeric_limits<T>::epsilon();
    if (std::abs(targetSize[0]) <= epsilon && std::abs(targetSize[1]) <= epsilon && std::abs(targetSize[2]) <= epsilon)
    {
        // if the target is empty, we return no overlap
        std::cout << "[findCollisions] empty target -> no overlap" << std::endl;
        return;
    }
    auto overlaps = [&](TreeNodeIndex idx)
    {
        auto [nk1, nk2] = decodePlaceholderBit2K(nodePrefixes[idx]);
        // std::cout << "[findCollisions] idx: " << idx << " nodeKey: " << std::oct << nk1 << ", " << nk2 << std::dec << std::endl;
        // std::cout << "[findCollisions] idx: " << idx << " nodeCenters: " << nodeCenters[idx][0] << ", " << nodeCenters[idx][1] << ", " << nodeCenters[idx][2] << std::endl;
        // std::cout << "[findCollisions] idx: " << idx << " nodeSizes: " << nodeSizes[idx][0] << ", " << nodeSizes[idx][1] << ", " << nodeSizes[idx][2] << std::endl;
        // std::cout << "[findCollisions] idx: " << idx << " targetCenter: " << targetCenter[0] << ", " << targetCenter[1] << ", " << targetCenter[2] << std::endl;
        // std::cout << "[findCollisions] idx: " << idx << " targetSize: " << targetSize[0] << ", " << targetSize[1] << ", " << targetSize[2] << std::endl;
        // std::cout << "[findCollisions] idx: " << idx << " !containedIn(nk1, nk2, excludeStart, excludeEnd): " << !containedIn(nk1, nk2, excludeStart, excludeEnd) << std::endl;
        // std::cout << "[findCollisions] idx: " << idx << " overlap: " << overlap(nodeCenters[idx], nodeSizes[idx], targetCenter, targetSize, box) << std::endl;
        constexpr T epsilon = std::numeric_limits<T>::epsilon();
        if (std::abs(nodeSizes[idx][0]) <= epsilon && std::abs(nodeSizes[idx][1]) <= epsilon && std::abs(nodeSizes[idx][2]) <= epsilon)
        {
            // if the cell is empty, we return no overlap
            // std::cout << "[findCollisions] idx: " << idx << " empty cell -> no overlap" << std::endl;
            return false;
        }
        // return false;

        bool bOverlap   = !containedIn(nk1, nk2, excludeStart, excludeEnd) &&
                        overlap(nodeCenters[idx], nodeSizes[idx], targetCenter, targetSize, box);
        if (bOverlap) {
            // std::cout << "idx " << idx << " box [ " << nodeCenters[idx][0] - nodeSizes[idx][0] << ", " << nodeCenters[idx][1] - nodeSizes[idx][1] << ", " << nodeCenters[idx][2] - nodeSizes[idx][2]
            //           << "] - [" << nodeCenters[idx][0] + nodeSizes[idx][0] << ", " << nodeCenters[idx][1] + nodeSizes[idx][1] << ", " << nodeCenters[idx][2] + nodeSizes[idx][2]
            //           << "] overlaps with target box [ " << targetCenter[0] - targetSize[0] << ", " << targetCenter[1] - targetSize[1] << ", " << targetCenter[2] - targetSize[2]
            //           << "] - [" << targetCenter[0] + targetSize[0] << ", " << targetCenter[1] + targetSize[1] << ", " << targetCenter[2] + targetSize[2]
            //           << "] nk1: " << std::oct << nk1 << " nk2: " << nk2 << std::dec << std::endl;
            flags[idx] = 1;
        }
        return bOverlap;
    };

    singleTraversal(childOffsets, parents, overlaps, [](TreeNodeIndex) {});
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
    // if (!use_mixD) {
    //     throw std::runtime_error("findHalos: non-mixD case not implemented");
    // }


#pragma omp parallel for
    for (TreeNodeIndex leafIdx = firstNode; leafIdx < lastNode; ++leafIdx)
    {
        constexpr Tc epsilon = std::numeric_limits<Tc>::epsilon();
        if (std::abs(searchSizes[leafIdx][0]) <= epsilon &&
            std::abs(searchSizes[leafIdx][1]) <= epsilon &&
            std::abs(searchSizes[leafIdx][2]) <= epsilon)
        {
            // if the target is empty, we skip it
            // std::cout << "[findHalos] Skipping leafIdx: " << leafIdx << std::endl;
            continue;
        }
        // std::cout << "[findHalos] leafIdx: " << leafIdx << " searchCenter: " << searchCenters[leafIdx][0] << ", " << searchCenters[leafIdx][1] << ", " << searchCenters[leafIdx][2] << std::endl;
        // std::cout << "[findHalos] leafIdx: " << leafIdx << " searchSize: " << searchSizes[leafIdx][0] << ", " << searchSizes[leafIdx][1] << ", " << searchSizes[leafIdx][2] << std::endl;
        // if the halo box is fully inside the assigned SFC range, we skip collision detection
        if (use_mixD && containedIn(lowestKey, highestKey, searchCenters[leafIdx], searchSizes[leafIdx], box, mixDBits.bx, mixDBits.by, mixDBits.bz)) { /*std::cout << "[findHalos] contained in TRUE" << std::endl;*/ continue; }
        if (!use_mixD && containedIn(lowestKey, highestKey, searchCenters[leafIdx], searchSizes[leafIdx], box)) { continue; }
        findCollisions(prefixes, childOffsets, parents, nodeCenters, nodeSizes, searchCenters[leafIdx],
                       searchSizes[leafIdx], box, lowestKey, highestKey, collisionFlags);
    }
}

} // namespace cstone
