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
IBox makeLevelBox(unsigned ix, unsigned iy, unsigned iz, unsigned level, AxesBits axesBits)
{
    unsigned Lx = 1u << std::min(maxTreeLevel<KeyType>{} - level, axesBits[0]);
    unsigned Ly = 1u << std::min(maxTreeLevel<KeyType>{} - level, axesBits[1]);
    unsigned Lz = 1u << std::min(maxTreeLevel<KeyType>{} - level, axesBits[2]);
    return IBox(ix * Lx, ix * Lx + Lx, iy * Ly, iy * Ly + Ly, iz * Lz, iz * Lz + Lz);
}

template<class KeyType>
constexpr AxesBits uniformAxesBits{maxTreeLevel<KeyType>{}, maxTreeLevel<KeyType>{}, maxTreeLevel<KeyType>{}};

template<class KeyType>
std::vector<IBox> makeUniformReference(unsigned level)
{
    const AxesBits axesBits = uniformAxesBits<KeyType>;
    return {
        makeLevelBox<KeyType>(0, 0, 0, level, axesBits), makeLevelBox<KeyType>(0, 0, 1, level, axesBits),
        makeLevelBox<KeyType>(0, 1, 0, level, axesBits), makeLevelBox<KeyType>(0, 1, 1, level, axesBits),
        makeLevelBox<KeyType>(1, 0, 0, level, axesBits), makeLevelBox<KeyType>(1, 0, 1, level, axesBits),
        makeLevelBox<KeyType>(1, 1, 0, level, axesBits), makeLevelBox<KeyType>(1, 1, 1, level, axesBits),
        makeLevelBox<KeyType>(0, 0, 2, level, axesBits), makeLevelBox<KeyType>(0, 1, 2, level, axesBits),
        makeLevelBox<KeyType>(1, 0, 2, level, axesBits), makeLevelBox<KeyType>(1, 1, 2, level, axesBits),
    };
}

template<class KeyType>
void surfaceDetection(IBox targetBox, std::vector<IBox> reference, unsigned numLeaves)
{
    const auto axesBits = getBoxDimensionBits<int, KeyType, IBox>(targetBox);

    std::vector<KeyType> tree = makeUniformNLevelTree<KeyType>(numLeaves, 1);

    Octree<KeyType> fullTree;
    fullTree.update(tree.data(), nNodes(tree));

    std::vector<IBox> treeBoxes(fullTree.numTreeNodes());
    for (TreeNodeIndex i = 0; i < fullTree.numTreeNodes(); ++i)
    {
        treeBoxes[i] = sfcIBox(sfcKey(fullTree.codeStart(i)), fullTree.level(i), axesBits);
    }

    auto isSurface = [targetBox, bbox = Box<double>(0, 1), boxes = treeBoxes.data()](TreeNodeIndex idx)
    { return minDistanceSq<KeyType>(targetBox, boxes[idx], bbox) == 0.0; };

    std::vector<IBox> surfaceBoxes;
    auto saveBox = [&surfaceBoxes, &treeBoxes](TreeNodeIndex idx)
    {
        if (treeBoxes[idx] != IBox(0, 0, 0, 0, 0, 0)) { surfaceBoxes.push_back(treeBoxes[idx]); }
    };

    singleTraversal(fullTree.childOffsets().data(), fullTree.parents().data(), isSurface, saveBox);

    std::sort(begin(surfaceBoxes), end(surfaceBoxes));
    std::sort(begin(reference), end(reference));
    EXPECT_EQ(surfaceBoxes, reference);
}

TEST(Traversal, surfaceDetection)
{
    const unsigned uniformLevel = 2;
    surfaceDetection<unsigned>(makeLevelBox<unsigned>(0, 0, 1, uniformLevel, uniformAxesBits<unsigned>),
                               makeUniformReference<unsigned>(uniformLevel), 64);
    surfaceDetection<uint64_t>(makeLevelBox<uint64_t>(0, 0, 1, uniformLevel, uniformAxesBits<uint64_t>),
                               makeUniformReference<uint64_t>(uniformLevel), 64);

    const IBox nonUniformTargetBox{0, 512, 0, 8, 0, 2};
    const auto nonUniformAxesBitsU32 = getBoxDimensionBits<int, unsigned, IBox>(nonUniformTargetBox);
    const auto nonUniformAxesBitsU64 = getBoxDimensionBits<int, uint64_t, IBox>(nonUniformTargetBox);
    surfaceDetection<unsigned>(nonUniformTargetBox,
                               {makeLevelBox<unsigned>(0, 0, 0, 3, nonUniformAxesBitsU32),
                                makeLevelBox<unsigned>(1, 0, 0, 3, nonUniformAxesBitsU32),
                                makeLevelBox<unsigned>(2, 0, 0, 3, nonUniformAxesBitsU32),
                                makeLevelBox<unsigned>(3, 0, 0, 3, nonUniformAxesBitsU32),
                                makeLevelBox<unsigned>(4, 0, 0, 3, nonUniformAxesBitsU32)},
                               256);
    surfaceDetection<uint64_t>(nonUniformTargetBox, {makeLevelBox<uint64_t>(0, 0, 0, 3, nonUniformAxesBitsU64)}, 256);
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
void dualTraversalNeighbors(const Box<float>& box, unsigned expectedPairs, unsigned expectedMultipolePairs)
{
    Octree<KeyType> octree;
    auto leaves = makeUniformNLevelTree<KeyType>(64, 1);
    octree.update(leaves.data(), nNodes(leaves));

    const auto axesBits = getBoxDimensionBits<float, KeyType, Box<float>>(box);

    for (TreeNodeIndex i = 0; i < octree.numTreeNodes(); ++i)
    {
        const auto box = sfcIBox(sfcKey(octree.codeStart(i)), octree.level(i), axesBits);
        if (box == IBox(0, 0, 0, 0, 0, 0)) { continue; }
    }

    KeyType focusStart = octree.codeStart(octree.toInternal(0));
    KeyType focusEnd   = octree.codeStart(octree.toInternal(8));

    auto crossFocusSurfacePairs =
        [focusStart, focusEnd, &tree = octree, &box, &axesBits](TreeNodeIndex a, TreeNodeIndex b)
    {
        bool aFocusOverlap = overlapTwoRanges(focusStart, focusEnd, tree.codeStart(a), tree.codeEnd(a));
        bool bInFocus      = containedIn(tree.codeStart(b), tree.codeEnd(b), focusStart, focusEnd);
        if (!aFocusOverlap || bInFocus) { return false; }

        IBox aBox = sfcIBox(sfcKey(tree.codeStart(a)), tree.level(a), axesBits);
        IBox bBox = sfcIBox(sfcKey(tree.codeStart(b)), tree.level(b), axesBits);
        const auto distance{minDistanceSq<KeyType>(aBox, bBox, box)};
        return std::abs(distance) < 1e-6;
    };

    std::vector<util::array<TreeNodeIndex, 2>> peer_pairs;
    auto p2p = [&peer_pairs](TreeNodeIndex a, TreeNodeIndex b) { peer_pairs.push_back({a, b}); };

    std::vector<util::array<TreeNodeIndex, 2>> multipole_pairs;
    auto m2l = [&multipole_pairs, &tree = octree, &axesBits, &box](TreeNodeIndex a, TreeNodeIndex b)
    {
        IBox aBox = sfcIBox(sfcKey(tree.codeStart(a)), tree.level(a), axesBits);
        IBox bBox = sfcIBox(sfcKey(tree.codeStart(b)), tree.level(b), axesBits);
        const auto distance{minDistanceSq<KeyType>(aBox, bBox, box)};
        if (std::abs(distance) < 1e-6) { multipole_pairs.push_back({a, b}); }
    };

    dualTraversal(octree.childOffsets().data(), 0, 0, crossFocusSurfacePairs, m2l, p2p);

    EXPECT_EQ(peer_pairs.size(), expectedPairs);
    std::sort(begin(peer_pairs), end(peer_pairs));
    for (auto p : peer_pairs)
    {
        auto a = p[0];
        auto b = p[1];
        // a in focus
        EXPECT_TRUE(octree.codeStart(a) >= focusStart && octree.codeEnd(a) <= focusEnd);
        // b outside focus
        EXPECT_TRUE(octree.codeStart(b) >= focusEnd || octree.codeEnd(a) <= focusStart);
        // a and be touch each other
        IBox aBox = sfcIBox(sfcKey(octree.codeStart(a)), octree.level(a), axesBits);
        IBox bBox = sfcIBox(sfcKey(octree.codeStart(b)), octree.level(b), axesBits);
        EXPECT_FLOAT_EQ((minDistanceSq<KeyType>(aBox, bBox, box)), 0.0);
    }
    EXPECT_EQ(multipole_pairs.size(), expectedMultipolePairs);
}

TEST(Traversal, dualTraversalNeighbors)
{
    dualTraversalNeighbors<unsigned>({0, 1}, 61, 50);
    dualTraversalNeighbors<uint64_t>({0, 1}, 61, 50);
    dualTraversalNeighbors<unsigned>({0, 1, 0, 0.015625, 0, 0.00390625}, 1, 2);
    dualTraversalNeighbors<uint64_t>({0, 1, 0, 0.015625, 0, 0.00390625}, 1, 2);
}

} // namespace cstone
