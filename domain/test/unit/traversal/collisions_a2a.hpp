/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Naive collision detection implementation for validation and testing
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/traversal/boxoverlap.hpp"
#include "cstone/tree/cs_util.hpp"

namespace cstone
{
/*! @brief to-all implementation of findCollisions
 *
 * @tparam     KeyType        32- or 64-bit unsigned integer
 * @param[in]  tree           octree leaf nodes in cornerstone format
 * @param[out] collisionList  output list of indices of colliding nodes
 * @param[in]  collisionBox   query box to look for collisions
 *                            with leaf nodes
 *
 * Naive implementation without tree traversal for reference
 * and testing purposes
 */
template<class KeyType, class T>
void findCollisions2All(std::span<const KeyType> nodeKeys,
                        const Vec3<T>* nodeCenters,
                        const Vec3<T>* nodeSizes,
                        const Box<T>& box,
                        Vec3<T> targetCenter,
                        Vec3<T> targetSize,
                        std::vector<TreeNodeIndex>& collisionList)
{
    for (TreeNodeIndex idx = 0; idx < TreeNodeIndex(nodeKeys.size()); ++idx)
    {
        if (norm2(minDistance(targetCenter, targetSize, nodeCenters[idx], nodeSizes[idx], box)) == 0.0)
        {
            collisionList.push_back(idx);
        }
    }
}

template<class KeyType>
void findCollisions2All(std::span<const KeyType> tree,
                        std::vector<TreeNodeIndex>& collisionList,
                        const IBox& collisionBox,
                        unsigned bx, unsigned by, unsigned bz)
{
    for (TreeNodeIndex idx = 0; idx < TreeNodeIndex(nNodes(tree)); ++idx)
    {
        IBox nodeBox = sfcIBox(sfcMixDKey(tree[idx]), sfcMixDKey(tree[idx + 1]), bx, by, bz);
        if (nodeBox.xmax() - nodeBox.xmin() == 0 && nodeBox.ymax() - nodeBox.ymin() == 0 &&
            nodeBox.zmax() - nodeBox.zmin() == 0)
        {
            // if the node is empty, we skip it
            // std::cout << "[findCollisions2All] Skipping idx: " << idx << " due to zero size" << std::endl;
            continue;
        }
        if (overlap<KeyType>(nodeBox, collisionBox)) { collisionList.push_back(idx); }
    }
}

//! @brief all-to-all implementation of findAllCollisions
template<class KeyType, class T>
std::vector<std::vector<TreeNodeIndex>> findCollisionsAll2all(std::span<const KeyType> nodeKeys,
                                                              const Vec3<T>* tC,
                                                              const Vec3<T>* tS,
                                                              TreeNodeIndex numTargets,
                                                              const Box<T>& box)
{
    std::vector<Vec3<T>> nodeCenters(nodeKeys.size()), nodeSizes(nodeKeys.size());
    nodeFpCenters<KeyType>(nodeKeys, nodeCenters.data(), nodeSizes.data(), box);

    std::vector<std::vector<TreeNodeIndex>> collisions(numTargets);
    for (TreeNodeIndex i = 0; i < numTargets; ++i)
    {
        findCollisions2All<KeyType>(nodeKeys, nodeCenters.data(), nodeSizes.data(), box, tC[i], tS[i], collisions[i]);
    }

    return collisions;
}

} // namespace cstone