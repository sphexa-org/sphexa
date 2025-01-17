/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Halo discovery tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/traversal/collisions.hpp"
#include "cstone/tree/cs_util.hpp"

#include "collisions_a2a.hpp"

using namespace cstone;

template<class KeyType, class T>
std::vector<int> findHalosAll2All(std::span<const KeyType> tree,
                                  const std::vector<T>& haloRadii,
                                  const Box<T>& box,
                                  TreeNodeIndex firstNode,
                                  TreeNodeIndex lastNode)
{
    std::vector<int> flags(nNodes(tree));
    auto collisions = findCollisionsAll2all(tree, haloRadii, box);

    for (TreeNodeIndex i = firstNode; i < lastNode; ++i)
    {
        for (TreeNodeIndex cidx : collisions[i])
        {
            if (cidx < firstNode || cidx >= lastNode) { flags[cidx] = 1; }
        }
    }

    return flags;
}

template<class KeyType>
void findHalosFlags()
{
    std::vector<KeyType> tree = makeUniformNLevelTree<KeyType>(64, 1);

    Box<double> box(0, 1);

    // size of one node is 0.25^3
    std::vector<double> interactionRadii(nNodes(tree), 0.1);

    Octree<KeyType> octree;
    octree.update(tree.data(), nNodes(tree));

    {
        std::vector<int> collisionFlags(nNodes(tree), 0);
        findHalos(octree.nodeKeys().data(), octree.childOffsets().data(), octree.toLeafOrder().data(), tree.data(),
                  interactionRadii.data(), box, 0, 32, collisionFlags.data());

        std::vector<int> reference = findHalosAll2All<KeyType>(tree, interactionRadii, box, 0, 32);

        // consistency check: the surface of the first 32 nodes with the last 32 nodes is 16 nodes
        EXPECT_EQ(16, std::accumulate(collisionFlags.begin(), collisionFlags.end(), 0));
        EXPECT_EQ(collisionFlags, reference);
    }
    {
        std::vector<int> collisionFlags(nNodes(tree), 0);
        findHalos(octree.nodeKeys().data(), octree.childOffsets().data(), octree.toLeafOrder().data(), tree.data(),
                  interactionRadii.data(), box, 32, 64, collisionFlags.data());

        std::vector<int> reference = findHalosAll2All<KeyType>(tree, interactionRadii, box, 32, 64);

        // consistency check: the surface of the first 32 nodes with the last 32 nodes is 16 nodes
        EXPECT_EQ(16, std::accumulate(collisionFlags.begin(), collisionFlags.end(), 0));
        EXPECT_EQ(collisionFlags, reference);
    }
}

TEST(HaloDiscovery, findHalosFlags)
{
    findHalosFlags<unsigned>();
    findHalosFlags<uint64_t>();
}
