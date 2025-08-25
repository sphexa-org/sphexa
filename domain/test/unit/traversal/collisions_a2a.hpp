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
template<class KeyType>
void findCollisions2All(std::span<const KeyType> tree,
                        std::vector<TreeNodeIndex>& collisionList,
                        const IBox& collisionBox)
{
    for (TreeNodeIndex idx = 0; idx < TreeNodeIndex(nNodes(tree)); ++idx)
    {
        IBox nodeBox = sfcIBox(sfcKey(tree[idx]), sfcKey(tree[idx + 1]));
        if (overlap<KeyType>(nodeBox, collisionBox)) { collisionList.push_back(idx); }
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
std::vector<std::vector<TreeNodeIndex>>
findCollisionsAll2all(std::span<const KeyType> tree, const std::vector<T>& haloRadii, const Box<T>& globalBox)
{
    std::vector<std::vector<TreeNodeIndex>> collisions(tree.size() - 1);

    const auto mixDBits = getBoxMixDimensionBits<T, KeyType, Box<T>>(globalBox);
    const bool use_mixD = mixDBits.bx != maxTreeLevel<KeyType>{} ||
                    mixDBits.by != maxTreeLevel<KeyType>{} ||
                    mixDBits.bz != maxTreeLevel<KeyType>{};

    for (TreeNodeIndex leafIdx = 0; leafIdx < TreeNodeIndex(nNodes(tree)); ++leafIdx)
    {
        T radius = haloRadii[leafIdx];

        IBox haloBox = makeHaloBox(tree[leafIdx], tree[leafIdx + 1], radius, globalBox);
        if (haloBox.xmax() - haloBox.xmin() == 0 && haloBox.ymax() - haloBox.ymin() == 0 &&
            haloBox.zmax() - haloBox.zmin() == 0)
        {
            // if the halo box is empty, we skip it
            // std::cout << "[findCollisionsAll2all] Skipping leafIdx: " << leafIdx << " due to zero size" << std::endl;
            continue;
        }
        if (use_mixD)
        {
            findCollisions2All<KeyType>(tree, collisions[leafIdx], haloBox, mixDBits.bx, mixDBits.by, mixDBits.bz);
        } else {
            findCollisions2All<KeyType>(tree, collisions[leafIdx], haloBox);
        }
    }

    return collisions;
}

} // namespace cstone