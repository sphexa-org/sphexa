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

#include "unit/traversal/collisions_a2a.hpp"

using namespace cstone;

/*! @brief compare tree-traversal collision detection with the naive all-to-all algorithm
 *
 * @tparam KeyType     32- or 64-bit unsigned integer
 * @tparam T           float or double
 * @param  tree        cornerstone octree leaves
 * @param  haloRadii   floating point collision radius per octree leaf
 * @param  box         bounding box used to construct the octree
 *
 * This test goes through all leaf nodes of the input octree and computes
 * a list of all other leaves that overlap with the first one.
 * The computation is done with both the tree-traversal algorithm and the
 * naive all-to-all algorithm and the results are compared.
 */
template<class KeyType, class T>
static void generalCollisionTest(const std::vector<KeyType>& tree, const std::vector<T>& haloRadii, const Box<T>& box)
{
    Octree<KeyType> octree;
    octree.update(tree.data(), nNodes(tree));

    // tree traversal collision detection
    std::vector<std::vector<TreeNodeIndex>> collisions(nNodes(tree));
    for (std::size_t leafIdx = 0; leafIdx < nNodes(tree); ++leafIdx)
    {
        T radius     = haloRadii[leafIdx];
        IBox haloBox = makeHaloBox(tree[leafIdx], tree[leafIdx + 1], radius, box);

        auto storeCollisions = [&collisionList = collisions[leafIdx], toLeaf = octree.toLeafOrder()](TreeNodeIndex i)
        { collisionList.push_back(toLeaf[i]); };

        findCollisions(octree.nodeKeys().data(), octree.childOffsets().data(), storeCollisions, haloBox, KeyType(0),
                       KeyType(0));
    }

    // naive all-to-all algorithm
    std::vector<std::vector<TreeNodeIndex>> refCollisions = findCollisionsAll2all<KeyType>(tree, haloRadii, box);

    for (std::size_t nodeIndex = 0; nodeIndex < nNodes(tree); ++nodeIndex)
    {
        std::sort(begin(collisions[nodeIndex]), end(collisions[nodeIndex]));
        std::sort(begin(refCollisions[nodeIndex]), end(refCollisions[nodeIndex]));

        EXPECT_EQ(collisions[nodeIndex], refCollisions[nodeIndex]);
    }
}

//! @brief an irregular tree with level-3 nodes next to level-1 ones
template<class I, class T, BoundaryType Pbc>
void irregularTreeTraversal()
{
    auto tree = OctreeMaker<I>{}.divide().divide(0).divide(0, 7).makeTree();

    Box<T> box(0, 1, 0, 1, 0, 1, Pbc, Pbc, Pbc);
    std::vector<T> haloRadii(nNodes(tree), 0.1);
    generalCollisionTest(tree, haloRadii, box);
}

TEST(Collisions, irregularTreeTraversal)
{
    irregularTreeTraversal<unsigned, float, BoundaryType::open>();
    irregularTreeTraversal<uint64_t, float, BoundaryType::open>();
    irregularTreeTraversal<unsigned, double, BoundaryType::open>();
    irregularTreeTraversal<uint64_t, double, BoundaryType::open>();
}

TEST(Collisions, irregularTreeTraversalPbc)
{
    irregularTreeTraversal<unsigned, float, BoundaryType::periodic>();
    irregularTreeTraversal<uint64_t, float, BoundaryType::periodic>();
    irregularTreeTraversal<unsigned, double, BoundaryType::periodic>();
    irregularTreeTraversal<uint64_t, double, BoundaryType::periodic>();
}

//! @brief a regular tree with level-3 nodes, 8x8x8 grid
template<class I, class T, BoundaryType Pbc>
void regularTreeTraversal()
{
    auto tree = makeUniformNLevelTree<I>(512, 1);

    Box<T> box(0, 1, 0, 1, 0, 1, Pbc, Pbc, Pbc);
    // node edge length is 0.125
    std::vector<T> haloRadii(nNodes(tree), 0.124);
    generalCollisionTest(tree, haloRadii, box);
}

TEST(Collisions, regularTreeTraversal)
{
    regularTreeTraversal<unsigned, float, BoundaryType::open>();
    regularTreeTraversal<uint64_t, float, BoundaryType::open>();
    regularTreeTraversal<unsigned, double, BoundaryType::open>();
    regularTreeTraversal<uint64_t, double, BoundaryType::open>();
}

TEST(Collisions, regularTreeTraversalPbc)
{
    regularTreeTraversal<unsigned, float, BoundaryType::periodic>();
    regularTreeTraversal<uint64_t, float, BoundaryType::periodic>();
    regularTreeTraversal<unsigned, double, BoundaryType::periodic>();
    regularTreeTraversal<uint64_t, double, BoundaryType::periodic>();
}

/*! @brief test tree traversal with anisotropic boxes
 *
 * anisotropic boxes with a single halo radius per node
 * results in different x,y,z halo search lengths once
 * the coordinates are normalized to the cubic unit box.
 */
class AnisotropicBoxTraversal : public testing::TestWithParam<std::array<int, 6>>
{
public:
    template<class I, class T>
    void check()
    {
        // 8x8x8 grid
        auto tree = makeUniformNLevelTree<I>(512, 1);

        Box<T> box(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()), std::get<3>(GetParam()),
                   std::get<4>(GetParam()), std::get<5>(GetParam()));

        // node edge length is 0.125 in the compressed dimension
        // and 0.250 in the other two dimensions
        std::vector<T> haloRadii(nNodes(tree), 0.175);
        generalCollisionTest(tree, haloRadii, box);
    }
};

TEST_P(AnisotropicBoxTraversal, compressedAxis32f) { check<unsigned, float>(); }

TEST_P(AnisotropicBoxTraversal, compressedAxis64f) { check<uint64_t, float>(); }

TEST_P(AnisotropicBoxTraversal, compressedAxis32d) { check<unsigned, double>(); }

TEST_P(AnisotropicBoxTraversal, compressedAxis64d) { check<uint64_t, double>(); }

std::vector<std::array<int, 6>> boxLimits{{0, 1, 0, 2, 0, 2}, {0, 2, 0, 1, 0, 2}, {0, 2, 0, 2, 0, 1}};

INSTANTIATE_TEST_SUITE_P(AnisotropicBoxTraversal, AnisotropicBoxTraversal, testing::ValuesIn(boxLimits));
