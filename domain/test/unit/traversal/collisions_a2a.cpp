/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Octree traversal tests with naive all-to-all collisions as reference
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/traversal/collisions.hpp"
#include "cstone/tree/cs_util.hpp"

#include "coord_samples/random.hpp"
#include "unit/traversal/collisions_a2a.hpp"

using namespace cstone;

/*! @brief compare tree-traversal collision detection with the naive all-to-all algorithm
 *
 * @tparam KeyType     32- or 64-bit unsigned integer
 * @tparam T           float or double
 * @param  octree        cornerstone octree leaves
 * @param  haloRadii   floating point collision radius per octree leaf
 * @param  box         bounding box used to construct the octree
 *
 * This test goes through all leaf nodes of the input octree and computes
 * a list of all other leaves that overlap with the first one.
 * The computation is done with both the tree-traversal algorithm and the
 * naive all-to-all algorithm and the results are compared.
 */
template<class KeyType, class T>
void generalCollisionTest(OctreeView<const KeyType> octree, const std::vector<T>& haloRadii, const Box<T>& box)
{
    std::vector<Vec3<T>> nodeCenters(octree.numNodes), nodeSizes(octree.numNodes);
    nodeFpCenters<KeyType>({octree.prefixes, size_t(octree.numNodes)}, nodeCenters.data(), nodeSizes.data(), box);

    auto leaf2int = octree.leafToInternalSpan();
    std::vector<Vec3<T>> tC(octree.numLeafNodes), tS(octree.numLeafNodes);
    for (size_t i = 0; i < size_t(octree.numLeafNodes); ++i)
    {
        tC[i] = nodeCenters[leaf2int[i]];
        tS[i] = nodeSizes[leaf2int[i]] * haloRadii[i];
    }

    // tree traversal collision detection
    std::vector<std::vector<TreeNodeIndex>> collisions(octree.numNodes);
    for (TreeNodeIndex i = 0; i < octree.numLeafNodes; ++i)
    {
        std::vector<uint8_t> flags(octree.numNodes, 0);
        findCollisions(octree.prefixes, octree.childOffsets, octree.parents, nodeCenters.data(), nodeSizes.data(),
                       tC[i], tS[i], box, KeyType(0), KeyType(0), flags.data());

        for (std::size_t fi = 0; fi < flags.size(); ++fi)
        {
            if (flags[fi]) { collisions[i].push_back(fi); }
        }
    }

    // naive all-to-all algorithm
    auto refCollisions = findCollisionsAll2all<KeyType>({octree.prefixes, size_t(octree.numNodes)}, tC.data(),
                                                        tS.data(), octree.numLeafNodes, box);

    for (TreeNodeIndex i = 0; i < octree.numLeafNodes; ++i)
    {
        std::ranges::sort(begin(collisions[i]), end(collisions[i]));
        std::ranges::sort(begin(refCollisions[i]), end(refCollisions[i]));

        EXPECT_EQ(collisions[i], refCollisions[i]);
    }
}

/*! @brief test tree traversal with anisotropic boxes
 *
 * Boxes with a single halo radius per node
 * results in different x,y,z halo search lengths once
 * the coordinates are normalized to the cubic unit box.
 */
class CollisionTests : public testing::TestWithParam<std::array<int, 4>>
{
public:
    template<class KeyType, class T>
    void check()
    {
        int numParticles          = 1000;
        std::vector<KeyType> keys = makeRandomGaussianKeys<KeyType>(numParticles);
        auto [tree, counts]       = computeOctree<KeyType>(keys, 4);

        OctreeData<KeyType, CpuTag> octree;
        octree.resize(nNodes(tree));
        updateInternalTree<KeyType>(tree, octree.data());

        auto bType = static_cast<BoundaryType>(std::get<3>(GetParam()));
        Box<T> box(0, std::get<0>(GetParam()), 0, std::get<1>(GetParam()), 0, std::get<2>(GetParam()), bType);

        std::vector<T> haloRadii(octree.numLeafNodes, 1.001);
        generalCollisionTest(octree.cdata(), haloRadii, box);
    }
};

TEST_P(CollisionTests, uint32f) { check<unsigned, float>(); }
TEST_P(CollisionTests, uint64d) { check<uint64_t, double>(); }

std::vector<std::array<int, 4>> boxLimits{{1, 2, 2, 0}, {2, 1, 2, 0}, {2, 2, 1, 0},
                                          {1, 2, 2, 1}, {2, 1, 2, 1}, {2, 2, 1, 1}};

INSTANTIATE_TEST_SUITE_P(CollisionTests, CollisionTests, testing::ValuesIn(boxLimits));
