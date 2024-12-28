/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Octree traversal tests with explicit references
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>

#include "gtest/gtest.h"

#include "cstone/traversal/collisions.hpp"
#include "cstone/tree/csarray.hpp"
#include "cstone/tree/cs_util.hpp"

using namespace cstone;

template<class KeyType>
static IBox makeLevelBox(unsigned ix, unsigned iy, unsigned iz, unsigned level)
{
    unsigned L = 1u << (maxTreeLevel<KeyType>{} - level);
    return IBox(ix * L, ix * L + L, iy * L, iy * L + L, iz * L, iz * L + L);
}

template<class KeyType>
static std::vector<TreeNodeIndex>
findCollidingIndices(IBox target, gsl::span<const KeyType> leaves, KeyType exclStart, KeyType exclEnd)
{
    Octree<KeyType> octree;
    octree.update(leaves.data(), nNodes(leaves));

    std::vector<TreeNodeIndex> collisions;
    auto storeCollisions = [toLeaf = octree.toLeafOrder().data(), &collisions](TreeNodeIndex i)
    { collisions.push_back(toLeaf[i]); };

    findCollisions(octree.nodeKeys().data(), octree.childOffsets().data(), storeCollisions, target, exclStart, exclEnd);
    std::sort(collisions.begin(), collisions.end());

    return collisions;
}

template<class KeyType>
static std::vector<IBox>
findCollidingBoxes(IBox target, gsl::span<const KeyType> leaves, KeyType exclStart, KeyType exclEnd)
{
    std::vector<TreeNodeIndex> collisions = findCollidingIndices(target, leaves, exclStart, exclEnd);

    std::vector<IBox> collidedNodeBoxes;
    for (auto cidx : collisions)
    {
        collidedNodeBoxes.push_back(sfcIBox(sfcKey(leaves[cidx]), sfcKey(leaves[cidx + 1])));
    }
    std::sort(begin(collidedNodeBoxes), end(collidedNodeBoxes));

    return collidedNodeBoxes;
}

/*! @brief test findCollisions with pbc halo boxes
 *
 * The example constructs a 4x4x4 regular octree with 64 nodes. A haloBox
 * with coordinates [-1,1]^3 will collide with all 8 nodes in the corners of the tree.
 * The same happens for a haloBox at the opposite diagonal end with
 * coordinates [2^(10 or 21)-1, 2^(10 or 21)+1]^3.
 */
template<class KeyType>
static void pbcCollision()
{
    std::vector<KeyType> tree = makeUniformNLevelTree<KeyType>(64, 1);

    // halo box around coordinate origin
    IBox haloBox{-1, 1, -1, 1, -1, 1};

    auto probe = findCollidingBoxes<KeyType>(haloBox, tree, 0, 0);

    constexpr unsigned L2 = 2;
    // all 8 corners collide
    std::vector<IBox> refCollided{
        makeLevelBox<KeyType>(0, 0, 0, L2), makeLevelBox<KeyType>(0, 0, 3, L2), makeLevelBox<KeyType>(0, 3, 0, L2),
        makeLevelBox<KeyType>(0, 3, 3, L2), makeLevelBox<KeyType>(3, 0, 0, L2), makeLevelBox<KeyType>(3, 0, 3, L2),
        makeLevelBox<KeyType>(3, 3, 0, L2), makeLevelBox<KeyType>(3, 3, 3, L2),
    };

    EXPECT_EQ(probe, refCollided);
}

TEST(OctreeTraversal, pbcCollision)
{
    pbcCollision<unsigned>();
    pbcCollision<uint64_t>();
}

/*! @brief test findCollisions with pbc halo boxes and part of the tree excluded from collision detection
 *
 * The example constructs a 4x4x4 regular octree with 64 nodes. A haloBox
 * with coordinates [-1,1]^3 will collide with all 8 nodes in the corners of the tree, except if excluded.
 * The same happens for a haloBox at the opposite diagonal end with
 * coordinates [2^(10 or 21)-1, 2^(10 or 21)+1]^3.
 */
template<class KeyType>
static void pbcCollisionWithExclusions()
{
    constexpr int L2          = 2;
    std::vector<KeyType> tree = makeUniformNLevelTree<KeyType>(64, 1);

    IBox haloBox{-1, 1, -1, 1, -1, 1};

    auto probe = findCollidingBoxes<KeyType>(haloBox, tree, 0, nodeRange<KeyType>(L2));

    // the (0,0,0) corner gets excluded
    std::vector<IBox> refCollided{
        makeLevelBox<KeyType>(0, 0, 3, L2), makeLevelBox<KeyType>(0, 3, 0, L2), makeLevelBox<KeyType>(0, 3, 3, L2),
        makeLevelBox<KeyType>(3, 0, 0, L2), makeLevelBox<KeyType>(3, 0, 3, L2), makeLevelBox<KeyType>(3, 3, 0, L2),
        makeLevelBox<KeyType>(3, 3, 3, L2),
    };

    EXPECT_EQ(probe, refCollided);
}

TEST(OctreeTraversal, pbcCollisionWithExclusions)
{
    pbcCollisionWithExclusions<unsigned>();
    pbcCollisionWithExclusions<uint64_t>();
}

/*! @brief test collision detection with anisotropic halo ranges
 *
 * If the bounding box of the floating point boundary box is not cubic,
 * an isotropic search range with one halo radius per node will correspond
 * to an anisotropic range in the Morton code SFC which always gets mapped
 * to an unit cube.
 */
template<class KeyType>
static void anisotropicHaloBox()
{
    std::vector<KeyType> tree = makeUniformNLevelTree<KeyType>(64, 1);

    int r = 1u << (maxTreeLevel<KeyType>{} - 2);

    // this will hit two nodes in +x direction, not just one neighbor node
    IBox haloBox(0, 4 * r, r, 2 * r, r, 2 * r);

    auto probe = findCollidingBoxes<KeyType>(haloBox, tree, 0, 0);

    constexpr unsigned L2 = 2;
    // the (0,0,0) corner gets excluded
    std::vector<IBox> refCollided{
        makeLevelBox<KeyType>(0, 1, 1, L2),
        makeLevelBox<KeyType>(1, 1, 1, L2),
        makeLevelBox<KeyType>(2, 1, 1, L2),
        makeLevelBox<KeyType>(3, 1, 1, L2),
    };

    EXPECT_EQ(probe, refCollided);
}

TEST(OctreeTraversal, anisotropicHalo)
{
    anisotropicHaloBox<unsigned>();
    anisotropicHaloBox<uint64_t>();
}

//! @brief this tree results from 2 particles at (0,0,0) and at (1,1,1) with a bucket size of 1
static std::vector<unsigned> makeEdgeTree()
{
    std::vector<unsigned> tree{
        0,          1,          2,          3,          4,          5,          6,          7,          8,
        16,         24,         32,         40,         48,         56,         64,         128,        192,
        256,        320,        384,        448,        512,        1024,       1536,       2048,       2560,
        3072,       3584,       4096,       8192,       12288,      16384,      20480,      24576,      28672,
        32768,      65536,      98304,      131072,     163840,     196608,     229376,     262144,     524288,
        786432,     1048576,    1310720,    1572864,    1835008,    2097152,    4194304,    6291456,    8388608,
        10485760,   12582912,   14680064,   16777216,   33554432,   50331648,   67108864,   83886080,   100663296,
        117440512,  134217728,  268435456,  402653184,  536870912,  671088640,  805306368,  939524096,  956301312,
        973078528,  989855744,  1006632960, 1023410176, 1040187392, 1056964608, 1059061760, 1061158912, 1063256064,
        1065353216, 1067450368, 1069547520, 1071644672, 1071906816, 1072168960, 1072431104, 1072693248, 1072955392,
        1073217536, 1073479680, 1073512448, 1073545216, 1073577984, 1073610752, 1073643520, 1073676288, 1073709056,
        1073713152, 1073717248, 1073721344, 1073725440, 1073729536, 1073733632, 1073737728, 1073738240, 1073738752,
        1073739264, 1073739776, 1073740288, 1073740800, 1073741312, 1073741376, 1073741440, 1073741504, 1073741568,
        1073741632, 1073741696, 1073741760, 1073741768, 1073741776, 1073741784, 1073741792, 1073741800, 1073741808,
        1073741816, 1073741817, 1073741818, 1073741819, 1073741820, 1073741821, 1073741822, 1073741823, 1073741824};

    return tree;
}

/*! @brief a simple collision test with the edge tree from above
 *
 * Since the halo radius for the first and last node is bigger than the box,
 * these two nodes collide with all nodes in the tree, while all other nodes have
 * radius 0 and only collide with themselves.
 */
TEST(Collisions, adjacentEdgeRegression)
{
    std::vector<unsigned> leaves = makeEdgeTree();

    Box<double> box(0.5, 0.6);

    std::vector<double> haloRadii(nNodes(leaves), 0);
    haloRadii[0]        = 0.2;
    *haloRadii.rbegin() = 0.2;

    std::vector<TreeNodeIndex> allNodes(nNodes(leaves));
    std::iota(begin(allNodes), end(allNodes), 0);

    for (std::size_t i = 0; i < nNodes(leaves); ++i)
    {
        IBox haloBox    = makeHaloBox(leaves[i], leaves[i + 1], haloRadii[i], box);
        auto collisions = findCollidingIndices<unsigned>(haloBox, leaves, 0u, 0u);

        if (i == 0 || i == nNodes(leaves) - 1) { EXPECT_EQ(collisions, allNodes); }
        else { EXPECT_EQ(collisions, std::vector<TreeNodeIndex>(1, i)); }
    }
}

/*! @brief collisions test with a very small radius
 *
 * This tests that a very small, but non-zero halo radius
 * does not get rounded down to zero.
 */
TEST(Collisions, adjacentEdgeSmallRadius)
{
    std::vector<unsigned> leaves = makeEdgeTree();
    Box<double> box(0, 1);

    // nNodes is 134
    TreeNodeIndex secondLastNode = 132;
    double radius                = 0.0001;
    IBox haloBox                 = makeHaloBox(leaves[secondLastNode], leaves[secondLastNode + 1], radius, box);

    auto collisions = findCollidingBoxes<unsigned>(haloBox, leaves, 0u, 0u);

    EXPECT_TRUE(std::unique(collisions.begin(), collisions.end()) == collisions.end());
    EXPECT_EQ(collisions.size(), 9);
    for (auto cbox : collisions)
    {
        EXPECT_TRUE(overlap<unsigned>(cbox, haloBox));
    }
}

TEST(Collisions, adjacentEdgeLastNode)
{
    std::vector<unsigned> leaves = makeEdgeTree();
    Box<double> box(0, 1);

    // nNodes is 134
    int lastNode  = 133;
    double radius = 0.0;
    IBox haloBox  = makeHaloBox(leaves[lastNode], leaves[lastNode + 1], radius, box);

    auto collisions = findCollidingIndices<unsigned>(haloBox, leaves, 0u, 0u);

    std::vector<TreeNodeIndex> refNodes{133};
    EXPECT_EQ(collisions, refNodes);
}
