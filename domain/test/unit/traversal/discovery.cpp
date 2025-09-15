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
                                      std::span<const TreeNodeIndex> leaf2int,
                                      const Vec3<T>* tC,
                                      const Vec3<T>* tS,
                                      TreeNodeIndex numTargets,
                                      const Box<T>& box,
                                      KeyType exclStart,
                                      KeyType exclEnd)
{
    std::vector<uint8_t> flags(nodeKeys.size());
    auto collisions = findCollisionsAll2all(nodeKeys, tC, tS, numTargets, box);

    for (size_t i = 0; i < collisions.size(); ++i)
    {
        auto [k1, k2] = decodePlaceholderBit2K(nodeKeys[leaf2int[i]]);
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
    using T = double;
    Box<T> box(0, 1);

    std::vector<KeyType> tree = makeUniformNLevelTree<KeyType>(64, 1);
    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(tree));
    updateInternalTree<KeyType>(tree, octree.data());

    std::vector<Vec3<T>> nodeCenters(octree.numNodes), nodeSizes(octree.numNodes);
    nodeFpCenters<KeyType>(octree.prefixes, nodeCenters.data(), nodeSizes.data(), box);
    auto leaf2int = leafToInternal(octree);

    // size of one node is 0.25^3
    std::vector<double> searchRadii(octree.numLeafNodes, 0.1);

    std::vector<Vec3<T>> tC(octree.numLeafNodes), tS(octree.numLeafNodes);
    for (size_t i = 0; i < size_t(octree.numLeafNodes); ++i)
    {
        tC[i] = nodeCenters[leaf2int[i]];
        tS[i] = nodeSizes[leaf2int[i]] + Vec3<T>{searchRadii[i], searchRadii[i], searchRadii[i]};
    }

    auto od = octree.data();
    {
        std::vector<uint8_t> collisionFlags(octree.numNodes, 0);
        findHalos(od.prefixes, od.childOffsets, od.parents, nodeCenters.data(), nodeSizes.data(), tree.data(),
                  tC.data(), tS.data(), box, 0, 32, collisionFlags.data());

        std::vector<uint8_t> reference = findHalosAll2All<KeyType>(octree.prefixes, leaf2int, tC.data(), tS.data(),
                                                                   octree.numLeafNodes, box, tree[0], tree[32]);

        // consistency check: the surface of the first 32 nodes with the last 32 nodes is 16 nodes (+5 internal nodes)
        EXPECT_EQ(21, std::accumulate(collisionFlags.begin(), collisionFlags.end(), 0));
        EXPECT_EQ(21, std::accumulate(reference.begin(), reference.end(), 0));
        EXPECT_EQ(collisionFlags, reference);
    }
    {
        std::vector<uint8_t> collisionFlags(octree.numNodes, 0);
        findHalos(od.prefixes, od.childOffsets, od.parents, nodeCenters.data(), nodeSizes.data(), tree.data(),
                  tC.data(), tS.data(), box, 32, 64, collisionFlags.data());

        std::vector<uint8_t> reference = findHalosAll2All<KeyType>(octree.prefixes, leaf2int, tC.data(), tS.data(),
                                                                   octree.numLeafNodes, box, tree[32], tree[64]);

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
