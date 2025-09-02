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
IBox makeLevelBox(unsigned ix, unsigned iy, unsigned iz, unsigned level)
{
    unsigned L = 1u << (maxTreeLevel<KeyType>{} - level);
    return IBox(ix * L, ix * L + L, iy * L, iy * L + L, iz * L, iz * L + L);
}

template<class KeyType>
void surfaceDetection()
{
    unsigned level            = 2;
    std::vector<KeyType> tree = makeUniformNLevelTree<KeyType>(64, 1);

    Octree<KeyType> fullTree;
    fullTree.update(tree.data(), nNodes(tree));

    IBox targetBox = makeLevelBox<KeyType>(0, 0, 1, level);

    std::vector<IBox> treeBoxes(fullTree.numTreeNodes());
    for (TreeNodeIndex i = 0; i < fullTree.numTreeNodes(); ++i)
    {
        treeBoxes[i] = sfcIBox(sfcKey(fullTree.codeStart(i)), fullTree.level(i));
    }

    auto isSurface = [targetBox, bbox = Box<double>(0, 1), boxes = treeBoxes.data()](TreeNodeIndex idx)
    {
        double distance = minDistanceSq<KeyType>(targetBox, boxes[idx], bbox);
        return distance == 0.0;
    };

    std::vector<IBox> surfaceBoxes;
    auto saveBox = [numInternalNodes = fullTree.numInternalNodes(), &surfaceBoxes, &treeBoxes](TreeNodeIndex idx)
    { surfaceBoxes.push_back(treeBoxes[idx]); };

    singleTraversal(fullTree.childOffsets().data(), fullTree.parents().data(), isSurface, saveBox);

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
    Octree<KeyType> fullTree;
    auto leaves = OctreeMaker<KeyType>{}.divide().divide(0).divide(0, 7).makeTree();
    fullTree.update(leaves.data(), nNodes(leaves));

    std::vector<util::array<TreeNodeIndex, 2>> pairs;

    auto allPairs = [](TreeNodeIndex, TreeNodeIndex) { return true; };

    auto m2l = [](TreeNodeIndex, TreeNodeIndex) {};
    auto p2p = [&pairs](TreeNodeIndex a, TreeNodeIndex b) { pairs.push_back({a, b}); };

    dualTraversal(fullTree.childOffsets().data(), 0, 0, allPairs, m2l, p2p);

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
    Octree<KeyType> octree;
    auto leaves = makeUniformNLevelTree<KeyType>(64, 1);
    octree.update(leaves.data(), nNodes(leaves));

    Box<float> box(0, 1);

    KeyType focusStart = octree.codeStart(octree.toInternal(0));
    KeyType focusEnd   = octree.codeStart(octree.toInternal(8));

    auto crossFocusSurfacePairs = [focusStart, focusEnd, &tree = octree, &box](TreeNodeIndex a, TreeNodeIndex b)
    {
        bool aFocusOverlap = overlapTwoRanges(focusStart, focusEnd, tree.codeStart(a), tree.codeEnd(a));
        bool bInFocus      = containedIn(tree.codeStart(b), tree.codeEnd(b), focusStart, focusEnd);
        if (!aFocusOverlap || bInFocus) { return false; }

        IBox aBox = sfcIBox(sfcKey(tree.codeStart(a)), tree.level(a));
        IBox bBox = sfcIBox(sfcKey(tree.codeStart(b)), tree.level(b));
        return minDistanceSq<KeyType>(aBox, bBox, box) == 0.0;
    };

    std::vector<util::array<TreeNodeIndex, 2>> pairs;
    auto p2p = [&pairs](TreeNodeIndex a, TreeNodeIndex b) { pairs.push_back({a, b}); };

    auto m2l = [](TreeNodeIndex, TreeNodeIndex) {};

    dualTraversal(octree.childOffsets().data(), 0, 0, crossFocusSurfacePairs, m2l, p2p);

    EXPECT_EQ(pairs.size(), 61);
    std::sort(begin(pairs), end(pairs));
    for (auto p : pairs)
    {
        auto a = p[0];
        auto b = p[1];
        // a in focus
        EXPECT_TRUE(octree.codeStart(a) >= focusStart && octree.codeEnd(a) <= focusEnd);
        // b outside focus
        EXPECT_TRUE(octree.codeStart(b) >= focusEnd || octree.codeEnd(a) <= focusStart);
        // a and be touch each other
        IBox aBox = sfcIBox(sfcKey(octree.codeStart(a)), octree.level(a));
        IBox bBox = sfcIBox(sfcKey(octree.codeStart(b)), octree.level(b));
        EXPECT_FLOAT_EQ((minDistanceSq<KeyType>(aBox, bBox, box)), 0.0);
    }
}

TEST(Traversal, dualTraversalNeighbors)
{
    dualTraversalNeighbors<unsigned>();
    dualTraversalNeighbors<uint64_t>();
}

} // namespace cstone