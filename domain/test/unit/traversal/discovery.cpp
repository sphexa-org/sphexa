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
std::vector<uint8_t> findHalosAll2All(std::span<const KeyType> nodeKeys,
                                      const std::vector<T>& haloRadii,
                                      const Box<T>& box,
                                      KeyType exclStart,
                                      KeyType exclEnd)
{
    std::vector<uint8_t> flags(nodeKeys.size());
    auto collisions = findCollisionsAll2all(nodeKeys, haloRadii, box);

    for (size_t i = 0; i < collisions.size(); ++i)
    {
        auto [k1, k2] = decodePlaceholderBit2K(nodeKeys[i]);
        if (!containedIn(k1, k2, exclStart, exclEnd)) { continue; } // select only targets in excluded range
        for (size_t cidx : collisions[i])
        {
            auto [k1, k2] = decodePlaceholderBit2K(nodeKeys[cidx]);
            if (!containedIn(k1, k2, exclStart, exclEnd)) { flags[cidx] = 1; } // only count sources outside exclusion
        }
    }

    return flags;
}

template<class KeyType>
void findHalosFlags()
{
    std::vector<KeyType> tree = makeUniformNLevelTree<KeyType>(64, 1);

    Box<double> box(0, 1);

    Octree<KeyType> octree;
    octree.update(tree.data(), nNodes(tree));
    auto leaf2int = octree.internalOrder();

    {
        // size of one node is 0.25^3
        std::vector<double> interactionRadii(nNodes(tree), 0.1);
        std::vector<double> iRadiiInt(octree.numTreeNodes(), 0.1);

        std::vector<uint8_t> collisionFlags(octree.numTreeNodes(), 0);
        findHalos(octree.nodeKeys().data(), octree.childOffsets().data(), octree.parents().data(), tree.data(),
                  interactionRadii.data(), box, 0, 32, collisionFlags.data());

        std::vector<uint8_t> reference =
            findHalosAll2All<KeyType>(octree.nodeKeys(), iRadiiInt, box, tree[0], tree[32]);

        // consistency check: the surface of the first 32 nodes with the last 32 nodes is 16 nodes
        EXPECT_EQ(21, std::accumulate(collisionFlags.begin(), collisionFlags.end(), 0));
        EXPECT_EQ(21, std::accumulate(reference.begin(), reference.end(), 0));
        EXPECT_EQ(collisionFlags, reference);
    }
    {
        // size of one node is 0.25^3
        std::vector<double> interactionRadii(nNodes(tree), 0.1);
        std::vector<double> iRadiiInt(octree.numTreeNodes(), 0.1);

        std::vector<uint8_t> collisionFlags(octree.numTreeNodes(), 0);
        findHalos(octree.nodeKeys().data(), octree.childOffsets().data(), octree.parents().data(), tree.data(),
                  interactionRadii.data(), box, 32, 64, collisionFlags.data());

        std::vector<uint8_t> reference =
            findHalosAll2All<KeyType>(octree.nodeKeys(), iRadiiInt, box, tree[32], tree[64]);

        // consistency check: the surface of the first 32 nodes with the last 32 nodes is 16 nodes
        EXPECT_EQ(21, std::accumulate(collisionFlags.begin(), collisionFlags.end(), 0));
        EXPECT_EQ(collisionFlags, reference);
    }
}

TEST(HaloDiscovery, findHalosFlags)
{
    findHalosFlags<unsigned>();
    findHalosFlags<uint64_t>();
}
