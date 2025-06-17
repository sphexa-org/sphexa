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
void findCollisions2All(std::span<const KeyType> nodeKeys,
                        std::vector<TreeNodeIndex>& collisionList,
                        const IBox& collisionBox)
{
    for (TreeNodeIndex idx = 0; idx < TreeNodeIndex(nodeKeys.size()); ++idx)
    {
        auto [k1, k2] = decodePlaceholderBit2K(nodeKeys[idx]);
        IBox nodeBox  = sfcIBox(sfcKey(k1), sfcKey(k2));
        if (overlap<KeyType>(nodeBox, collisionBox)) { collisionList.push_back(idx); }
    }
}

//! @brief all-to-all implementation of findAllCollisions
template<class KeyType, class T>
std::vector<std::vector<TreeNodeIndex>>
findCollisionsAll2all(std::span<const KeyType> nodeKeys, const std::vector<T>& haloRadii, const Box<T>& globalBox)
{
    std::vector<std::vector<TreeNodeIndex>> collisions(nodeKeys.size());

    for (TreeNodeIndex i = 0; i < TreeNodeIndex(nodeKeys.size()); ++i)
    {
        T radius = haloRadii[i];

        auto [k1, k2] = decodePlaceholderBit2K(nodeKeys[i]);
        IBox haloBox = makeHaloBox(k1, k2, radius, globalBox);
        findCollisions2All<KeyType>(nodeKeys, collisions[i], haloBox);
    }

    return collisions;
}

} // namespace cstone