/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Generic octree traversal tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/tree/cs_util.hpp"
#include "cstone/traversal/macs.hpp"
#include "cstone/traversal/traversal.hpp"

namespace cstone
{

template<class KeyType>
constexpr AxesBits uniformAxesBits{maxTreeLevel<KeyType>{}, maxTreeLevel<KeyType>{}, maxTreeLevel<KeyType>{}};

template<class KeyType>
IBox makeLevelBox(unsigned ix, unsigned iy, unsigned iz, unsigned level)
{
    unsigned L = 1u << (maxTreeLevel<KeyType>{} - level);
    return IBox(ix * L, ix * L + L, iy * L, iy * L + L, iz * L, iz * L + L);
}

template<class KeyType>
void surfaceDetection()
{
    unsigned level              = 2;
    std::vector<KeyType> leaves = makeUniformNLevelTree<KeyType>(64, 1);

    OctreeData<KeyType, execution::Cpu> octree;
    octree.resize(nNodes(leaves));
    updateInternalTree<KeyType>(leaves, octree.data());

    IBox targetBox = makeLevelBox<KeyType>(0, 0, 1, level);

    std::vector<IBox> treeBoxes(octree.numNodes);
    for (TreeNodeIndex i = 0; i < octree.numNodes; ++i)
    {
        auto [k1, k2] = decodePlaceholderBit2K(octree.prefixes[i]);
        treeBoxes[i]  = sfcIBox(sfcKey(k1), treeLevel(k2 - k1), uniformAxesBits<KeyType>);
    }

    auto isSurface = [targetBox, bbox = Box<double>(0, 1), boxes = treeBoxes.data()](TreeNodeIndex idx)
    {
        auto [aCenter, aSize] = centerAndSize<KeyType>(targetBox, bbox);
        auto [bCenter, bSize] = centerAndSize<KeyType>(boxes[idx], bbox);
        return norm2(minDistance(aCenter, aSize, bCenter, bSize, bbox)) == 0.0;
    };

    std::vector<IBox> surfaceBoxes;
    auto saveBox = [&surfaceBoxes, &treeBoxes](TreeNodeIndex idx) { surfaceBoxes.push_back(treeBoxes[idx]); };

    singleTraversal(octree.childOffsets.data(), octree.parents.data(), isSurface, saveBox);

    std::sort(begin(surfaceBoxes), end(surfaceBoxes));

    // Morton node indices at surface:  {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14};
    // Hilbert node indices at surface: {0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 15};

    // coordinates of 3D-node boxes that touch targetBox
    std::vector<IBox> reference{
        makeLevelBox<KeyType>(0, 0, 0, 2), makeLevelBox<KeyType>(0, 0, 1, 2), makeLevelBox<KeyType>(0, 1, 0, 2),
        makeLevelBox<KeyType>(0, 1, 1, 2), makeLevelBox<KeyType>(1, 0, 0, 2), makeLevelBox<KeyType>(1, 0, 1, 2),
        makeLevelBox<KeyType>(1, 1, 0, 2), makeLevelBox<KeyType>(1, 1, 1, 2), makeLevelBox<KeyType>(0, 0, 2, 2),
        makeLevelBox<KeyType>(0, 1, 2, 2), makeLevelBox<KeyType>(1, 0, 2, 2), makeLevelBox<KeyType>(1, 1, 2, 2),
    };

    std::sort(begin(reference), end(reference));
    EXPECT_EQ(surfaceBoxes, reference);
}

TEST(Traversal, surfaceDetection)
{
    surfaceDetection<unsigned>();
    surfaceDetection<uint64_t>();
}

//! @brief mac criterion refines all nodes, traverses the entire tree and finds all leaf-pairs
template<class KeyType>
void dualTraversalAllPairs()
{
    OctreeData<KeyType, execution::Cpu> octree;
    auto leaves = OctreeMaker<KeyType>{}.divide().divide(0).divide(0, 7).makeTree();
    octree.resize(nNodes(leaves));
    updateInternalTree<KeyType>(leaves, octree.data());

    std::vector<util::array<TreeNodeIndex, 2>> pairs;

    auto allPairs = [](TreeNodeIndex, TreeNodeIndex) { return true; };

    auto m2l = [](TreeNodeIndex, TreeNodeIndex) {};
    auto p2p = [&pairs](TreeNodeIndex a, TreeNodeIndex b) { pairs.push_back({a, b}); };

    dualTraversal(octree.childOffsets.data(), 0, 0, allPairs, m2l, p2p);

    std::sort(begin(pairs), end(pairs));
    auto uit = std::unique(begin(pairs), end(pairs));
    EXPECT_EQ(uit, end(pairs));
    EXPECT_EQ(pairs.size(), 484); // 22 leaves ^2 = 484
}

TEST(Traversal, dualTraversalAllPairs)
{
    dualTraversalAllPairs<unsigned>();
    dualTraversalAllPairs<uint64_t>();
}

/*! @brief dual traversal with A, B across a focus range and touching each other
 *
 * This finds all pairs of leaves (a,b) that touch each other and with
 * a inside the focus and b outside.
 */
template<class KeyType>
void dualTraversalNeighbors()
{
    OctreeData<KeyType, execution::Cpu> octree;
    auto leaves = makeUniformNLevelTree<KeyType>(64, 1);
    octree.resize(nNodes(leaves));
    updateInternalTree<KeyType>(leaves, octree.data());

    Box<float> box(0, 1);

    KeyType focusStart = leaves[0];
    KeyType focusEnd   = leaves[8];

    auto crossFocusSurfacePairs = [focusStart, focusEnd, &tree = octree, &box](TreeNodeIndex a, TreeNodeIndex b)
    {
        auto [ka1, ka2] = decodePlaceholderBit2K(tree.prefixes[a]);
        auto [kb1, kb2] = decodePlaceholderBit2K(tree.prefixes[b]);
        bool aFocusOverlap = overlapTwoRanges(focusStart, focusEnd, ka1, ka2);
        bool bInFocus      = containedIn(kb1, kb2, focusStart, focusEnd);
        if (!aFocusOverlap || bInFocus) { return false; }

        IBox aBox             = sfcIBox(sfcKey(ka1), treeLevel(ka2 - ka1), uniformAxesBits<KeyType>);
        IBox bBox             = sfcIBox(sfcKey(kb1), treeLevel(kb2 - kb1), uniformAxesBits<KeyType>);
        auto [aCenter, aSize] = centerAndSize<KeyType>(aBox, box);
        auto [bCenter, bSize] = centerAndSize<KeyType>(bBox, box);
        if (aSize == Vec3<float>{0, 0, 0} || bSize == Vec3<float>{0, 0, 0})
        {
            // if a or b has no size, it's from a unmapped area of a mix-dim SFC curve and contains no particles
            return false;
        }
        return norm2(minDistance(aCenter, aSize, bCenter, bSize, box)) == 0.0;
    };

    std::vector<util::array<TreeNodeIndex, 2>> pairs;
    auto p2p = [&pairs](TreeNodeIndex a, TreeNodeIndex b) { pairs.push_back({a, b}); };

    auto m2l = [](TreeNodeIndex, TreeNodeIndex) {};

    dualTraversal(octree.childOffsets.data(), 0, 0, crossFocusSurfacePairs, m2l, p2p);

    EXPECT_EQ(pairs.size(), 61);
    std::sort(begin(pairs), end(pairs));
    for (auto p : pairs)
    {
        auto [ka1, ka2] = decodePlaceholderBit2K(octree.prefixes[p[0]]);
        auto [kb1, kb2] = decodePlaceholderBit2K(octree.prefixes[p[1]]);
        // a in focus
        EXPECT_TRUE(ka1 >= focusStart && ka2 <= focusEnd);
        // b outside focus
        EXPECT_TRUE(kb1 >= focusEnd || kb2 <= focusStart);
        IBox aBox             = sfcIBox(sfcKey(ka1), nodeRange<KeyType>(ka2 - ka1), uniformAxesBits<KeyType>);
        IBox bBox             = sfcIBox(sfcKey(kb1), nodeRange<KeyType>(kb2 - kb1), uniformAxesBits<KeyType>);
        auto [aCenter, aSize] = centerAndSize<KeyType>(aBox, box);
        auto [bCenter, bSize] = centerAndSize<KeyType>(bBox, box);
        // a and be touch each other
        EXPECT_FLOAT_EQ(norm2(minDistance(aCenter, aSize, bCenter, bSize, box)), 0.0);
    }
}

TEST(Traversal, dualTraversalNeighbors)
{
    dualTraversalNeighbors<unsigned>();
    dualTraversalNeighbors<uint64_t>();
}

} // namespace cstone