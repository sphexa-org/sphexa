/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Test locally essential octree
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/focus/octree_focus.hpp"
#include "cstone/tree/cs_util.hpp"

namespace cstone
{

template<class KeyType>
static auto computeNodeOps(const OctreeView<KeyType>& octree,
                           const std::vector<unsigned>& leafCounts,
                           const std::vector<uint8_t>& csMacs,
                           KeyType focusStart,
                           KeyType focusEnd,
                           unsigned bucketSize,
                           const std::vector<std::tuple<KeyType, int>>& imacs)
{
    std::vector<unsigned> counts(octree.numNodes);

    std::span<const TreeNodeIndex> leafToInternal(octree.leafToInternal + octree.numInternalNodes, octree.numLeafNodes);
    std::span<const TreeNodeIndex> levelRange(octree.levelRange, maxTreeLevel<KeyType>{} + 2);

    scatter(leafToInternal, leafCounts.data(), counts.data());
    upsweep(levelRange, octree.childOffsets, counts.data(), NodeCount<unsigned>{});

    std::vector<uint8_t> macs(octree.numNodes);
    scatter(leafToInternal, csMacs.data(), macs.data());

    // transfer values for internal node macs
    for (auto t : imacs)
    {
        auto [key, value] = t;
        TreeNodeIndex idx = locateNode(key, octree.prefixes, octree.levelRange);
        macs[idx]         = value;
    }

    std::vector<int> nodeOps(octree.numNodes);
    rebalanceDecisionEssential({octree.prefixes, size_t(octree.numNodes)}, octree.childOffsets, octree.parents,
                               counts.data(), macs.data(), focusStart, focusEnd, bucketSize, nodeOps.data());
    bool converged =
        protectAncestors(std::span<const KeyType>(octree.prefixes, octree.numNodes), octree.parents, nodeOps.data());

    std::vector<int> ret(octree.numLeafNodes);
    gather(leafToInternal, nodeOps.data(), ret.data());

    return std::make_tuple(ret, converged);
}

//! @brief various tests about merge/split decisions based on node counts and MACs
template<class KeyType>
static void rebalanceDecision()
{
    unsigned bucketSize = 1;

    {
        std::vector<KeyType> cstree = OctreeMaker<KeyType>{}.divide().divide(0).divide(7).makeTree();
        OctreeData<KeyType, CpuTag> octree;
        octree.resize(nNodes(cstree));
        updateInternalTree<KeyType>(cstree, octree.data());

        // nodes 14-21 should be fused based on counts, and 14 should be split based on MACs. counts win, nodes are
        // fused
        //                               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
        std::vector<unsigned> leafCounts{1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
        std::vector<uint8_t> macs{/*     */ 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};

        std::vector<int> reference{1, 1, 1, 8, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};

        // internal cell macs in placeholder bit format
        std::vector<std::tuple<KeyType, int>> internalMacs{{1, 1}, {010, 1}, {017, 1}};
        auto [nodeOps, converged] =
            computeNodeOps(octree.data(), leafCounts, macs, cstree[0], cstree[8], bucketSize, internalMacs);

        EXPECT_EQ(nodeOps, reference);
        EXPECT_FALSE(converged);
    }
    {
        std::vector<KeyType> cstree = OctreeMaker<KeyType>{}.divide().divide(0).divide(7).makeTree();
        OctreeData<KeyType, CpuTag> octree;
        octree.resize(nNodes(cstree));
        updateInternalTree<KeyType>(cstree, octree.data());

        // nodes 14-21 should be split/stay based on counts, and should stay based on MACs.
        // MAC wins, nodes stay, but are not split
        //                               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
        std::vector<unsigned> leafCounts{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0};
        std::vector<uint8_t> macs{/*     */ 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
        //                            ^
        //                            parent of leaf nodes 14-21
        std::vector<int> reference{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

        std::vector<std::tuple<KeyType, int>> internalMacs{{1, 1}, {010, 1}, {017, 1}};
        auto [nodeOps, converged] =
            computeNodeOps(octree.data(), leafCounts, macs, cstree[0], cstree[8], bucketSize, internalMacs);

        EXPECT_EQ(nodeOps, reference);
        EXPECT_TRUE(converged);
    }
    {
        std::vector<KeyType> cstree = OctreeMaker<KeyType>{}.divide().divide(0).divide(7).makeTree();
        OctreeData<KeyType, CpuTag> octree;
        octree.resize(nNodes(cstree));
        updateInternalTree<KeyType>(cstree, octree.data());

        // nodes 14-21 should stay based on counts, and should be fused based on MACs. MAC wins, nodes are fused
        //                               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
        std::vector<unsigned> leafCounts{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0};
        std::vector<uint8_t> macs{/*     */ 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
        //                            ^
        //                            parent of leaf nodes 14-21
        std::vector<int> reference{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};

        std::vector<std::tuple<KeyType, int>> internalMacs{{1, 1}, {010, 1}, {017, 0}};
        auto [nodeOps, converged] =
            computeNodeOps(octree.data(), leafCounts, macs, cstree[0], cstree[8], bucketSize, internalMacs);

        EXPECT_EQ(nodeOps, reference);
        EXPECT_FALSE(converged);
    }
    {
        // this example has a focus area that cuts through sets of 8 neighboring sibling nodes
        std::vector<KeyType> cstree = OctreeMaker<KeyType>{}.divide().divide(0).divide(1).makeTree();
        OctreeData<KeyType, CpuTag> octree;
        octree.resize(nNodes(cstree));
        updateInternalTree<KeyType>(cstree, octree.data());

        //                               |                     |                       |
        //                               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
        std::vector<unsigned> leafCounts{1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1};
        std::vector<uint8_t> macs{/*     */ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
        //                 root ^  ^  ^
        //   parent of leaves 0-7  |  | parent of leaf nodes 8-15                  ^ here count says split, mac says
        //                                                                           merge, result: stay
        std::vector<int> reference{1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1};
        //                                                             ----------------
        //                   these nodes are kept alive because their siblings (8 and 9) are inside the focus and are
        //                   staying

        //                                               root  parent of leaves 0-7
        //                                                  |       |           . parent of leaves 8-15
        std::vector<std::tuple<KeyType, int>> internalMacs{{1, 1}, {010, 1}, {011, 0}};
        auto [nodeOps, converged] =
            computeNodeOps(octree.data(), leafCounts, macs, cstree[2], cstree[10], bucketSize, internalMacs);

        EXPECT_EQ(nodeOps, reference);
        EXPECT_FALSE(converged);
    }
    {
        std::vector<KeyType> cstree = OctreeMaker<KeyType>{}.divide().divide(6).divide(7).makeTree();
        OctreeData<KeyType, CpuTag> octree;
        octree.resize(nNodes(cstree));
        updateInternalTree<KeyType>(cstree, octree.data());

        //                                                                        focus
        //                               |                |                      |-----------------------
        //                               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
        std::vector<unsigned> leafCounts{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        std::vector<uint8_t> macs{/*     */ 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        std::vector<int> reference{1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
        // Check that nodes 6-13, directly adjacent to the focus area can be merged

        std::vector<std::tuple<KeyType, int>> internalMacs{{1, 1}, {016, 0}, {017, 0}};
        auto [nodeOps, converged] =
            computeNodeOps(octree.data(), leafCounts, macs, cstree[14], cstree[22], bucketSize, internalMacs);

        EXPECT_EQ(nodeOps, reference);
        EXPECT_FALSE(converged);
    }
}

TEST(FocusedOctree, rebalanceDecision)
{
    rebalanceDecision<unsigned>();
    rebalanceDecision<uint64_t>();
}

template<class KeyType>
static void nodeOpsKeepAlive()
{
    {
        std::vector<KeyType> cstree = OctreeMaker<KeyType>{}.divide().divide(0).divide(7).makeTree();
        OctreeData<KeyType, CpuTag> octree;
        octree.resize(nNodes(cstree));
        updateInternalTree<KeyType>(cstree, octree.data());

        // | 0 | 1 2 3 4 5 6 7 8 |
        //       |             |
        //       |             17 18 19 20 21 22 23 24
        //       9 10 11 12 13 14 15 16
        //                                 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
        std::vector<TreeNodeIndex> nodeOps{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        for (size_t i = 0; i < nodeOps.size(); ++i)
        {
            nodeOps[i] = nzAncestorOp(i, octree.prefixes.data(), octree.parents.data(), nodeOps.data());
        }

        //                                   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
        std::vector<TreeNodeIndex> reference{1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        EXPECT_EQ(nodeOps, reference);
    }
}

TEST(FocusedOctree, nodeOpsKeepAlive)
{
    nodeOpsKeepAlive<unsigned>();
    nodeOpsKeepAlive<uint64_t>();
}

TEST(FocusedOctree, keyEnforcement)
{
    using KeyType = unsigned;

    {
        auto cstree = OctreeMaker<KeyType>{}.divide().divide(1).makeTree();
        OctreeData<KeyType, CpuTag> octree_;
        octree_.resize(nNodes(cstree));
        updateInternalTree<KeyType>(cstree, octree_.data());
        auto octree = octree_.data();

        // | 0 | 1 2 3 4 5 6 7 8 |
        //         |
        //         |
        //         9 10 11 12 13 14 15 16
        {
            //                       0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
            std::vector<int> nodeOps{1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            auto status = enforceKeySingle(decodePlaceholderBit(0111u), octree.prefixes, octree.childOffsets,
                                           octree.parents, nodeOps.data());

            std::vector<int> ref{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
            EXPECT_EQ(status, ResolutionStatus::cancelMerge);
            EXPECT_EQ(nodeOps, ref);
        }
        {
            //                       0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
            std::vector<int> nodeOps{1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            auto status = enforceKeySingle(decodePlaceholderBit(01112u), octree.prefixes, octree.childOffsets,
                                           octree.parents, nodeOps.data());

            //                   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
            std::vector<int> ref{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1};
            EXPECT_EQ(status, ResolutionStatus::rebalance);
            EXPECT_EQ(nodeOps, ref);
        }
        {
            //                       0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
            std::vector<int> nodeOps{1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            auto status = enforceKeySingle(decodePlaceholderBit(0101u), octree.prefixes, octree.childOffsets,
                                           octree.parents, nodeOps.data());

            //                   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
            std::vector<int> ref{1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
            protectAncestors(std::span<const KeyType>(octree.prefixes, octree.numNodes), octree.parents,
                             nodeOps.data());
            EXPECT_EQ(status, ResolutionStatus::rebalance);
            EXPECT_EQ(nodeOps, ref);
        }
        {
            // this tests that the splitting of node 1 does not invalidate the merge of nodes 9-16
            //                       0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
            std::vector<int> nodeOps{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
            //                          ^ injection splits node 1
            auto status = enforceKeySingle(decodePlaceholderBit(0101u), octree.prefixes, octree.childOffsets,
                                           octree.parents, nodeOps.data());
            EXPECT_EQ(status, ResolutionStatus::rebalance);

            status = enforceKeySingle(decodePlaceholderBit(01011u), octree.prefixes, octree.childOffsets,
                                      octree.parents, nodeOps.data());
            EXPECT_EQ(status, ResolutionStatus::failed);

            std::vector<int> ref{1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
            EXPECT_EQ(nodeOps, ref);
        }
    }
}

template<class KeyType>
TreeNodeIndex numNodesInRange(std::span<const KeyType> tree, KeyType a, KeyType b)
{
    auto itb = std::lower_bound(tree.begin(), tree.end(), b);
    auto ita = std::lower_bound(tree.begin(), tree.end(), a);

    EXPECT_EQ(*itb, b);
    EXPECT_EQ(*ita, a);

    return itb - ita;
}

template<class KeyType>
std::vector<TreeNodeIndex> octantNodeCount(std::span<const KeyType> tree)
{
    std::vector<TreeNodeIndex> counts;
    for (int octant = 0; octant < 8; ++octant)
    {
        auto range = nodeRange<KeyType>(1);
        counts.push_back(numNodesInRange(tree, octant * range, octant * range + range));
    }

    std::sort(begin(counts), end(counts));

    // omit the last octant with the highest node count (the focused octant)
    return {counts.begin(), counts.begin() + 7};
}

template<class KeyType>
static void computeEssentialTree()
{
    Box<double> box{-1, 1};
    int nParticles        = 200000;
    unsigned csBucketSize = 16;

    std::vector<KeyType> keys(nParticles);
    std::iota(keys.begin(), keys.end(), 0);
    std::for_each(keys.begin(), keys.end(), [n = nParticles](auto& k) { k *= double(nodeRange<KeyType>(0)) / n; });

    auto [csTree, csCounts] = computeOctree<KeyType>(keys, csBucketSize);
    Octree<KeyType> globalTree;
    globalTree.update(csTree.data(), nNodes(csTree));

    unsigned bucketSize = 16;
    float theta         = 0.9;
    FocusedOctreeSingleNode<KeyType> tree(bucketSize, theta);

    // sorted reference tree node counts in each (except focus) octant at the 1st division level
    std::vector<TreeNodeIndex> refCounts{29, 302, 302, 302, 1184, 1184, 1184,
                                         /*4131*/};

    KeyType focusStart = 1;
    KeyType focusEnd   = pad(KeyType(1), 3);

    tree.update(box, keys, focusStart, focusEnd, {});
    // The focus boundaries have to be contained in the tree, even after just one update step.
    // This example here is the worst-case scenario with a focus boundary at the highest possible
    // octree subdivision level. Key-0 is always present, so the node with Key-1 is always at index 1, if present.
    EXPECT_EQ(tree.treeLeaves()[1], focusStart);

    // update until converged
    while (!tree.update(box, keys, focusStart, focusEnd, {})) {}

    {
        // the first node in the cornerstone tree that starts at or above focusStart
        TreeNodeIndex firstCstoneNode = findNodeAbove(csTree.data(), csTree.size(), focusStart);
        TreeNodeIndex matchingFocusNode =
            findNodeAbove(tree.treeLeaves().data(), tree.treeLeaves().size(), csTree[firstCstoneNode]);
        // in the focus area (the first octant) the essential tree and the csTree are identical
        TreeNodeIndex lastFocusNode = findNodeAbove(tree.treeLeaves().data(), tree.treeLeaves().size(), focusEnd);

        EXPECT_TRUE(matchingFocusNode >= 1 && matchingFocusNode < lastFocusNode);

        // We have: focusTree[matchingFocusNode] == csTree[firstCstoneNode]
        // therefore all nodes past matchingFocusNode until lastFocusNode should match those in the csTree
        EXPECT_TRUE(std::equal(tree.treeLeaves().begin() + matchingFocusNode, tree.treeLeaves().begin() + lastFocusNode,
                               begin(csTree) + firstCstoneNode));

        // From 0 to matchingFocusNode, the focusTree should be identical to the spanningTree
        std::vector<KeyType> spanningKeys{0, focusStart, nodeRange<KeyType>(0)};
        auto spanningTree = computeSpanningTree<KeyType>(spanningKeys);
        auto a            = tree.treeLeaves().first(matchingFocusNode);
        EXPECT_TRUE(std::equal(a.begin(), a.end(), spanningTree.data()));

        auto nodeCounts = octantNodeCount<KeyType>(tree.treeLeaves());
        EXPECT_EQ(nodeCounts, refCounts);
    }

    focusStart = pad(KeyType(7), 3);
    focusEnd   = nodeRange<KeyType>(0) - 1;
    while (!tree.update(box, keys, focusStart, focusEnd, {})) {}

    {
        auto nodeCounts = octantNodeCount<KeyType>(tree.treeLeaves());
        EXPECT_EQ(nodeCounts, refCounts);
    }

    focusStart = 0; // slight variation; start from zero instead of 1
    focusEnd   = pad(KeyType(1), 3);
    while (!tree.update(box, keys, focusStart, focusEnd, {})) {}

    {
        TreeNodeIndex lastFocusNode = findNodeAbove(tree.treeLeaves().data(), tree.treeLeaves().size(), focusEnd);
        // tree now focused again on first octant
        EXPECT_TRUE(std::equal(begin(csTree), begin(csTree) + lastFocusNode, tree.treeLeaves().begin()));

        auto nodeCounts = octantNodeCount<KeyType>(tree.treeLeaves());
        EXPECT_EQ(nodeCounts, refCounts);
    }
}

TEST(FocusedOctree, compute)
{
    computeEssentialTree<unsigned>();
    computeEssentialTree<uint64_t>();
}

class MacRefinement : public testing::Test
{
protected:
    using KeyType = uint64_t;
    using T       = double;

    void SetUp() override
    {
        OctreeMaker<KeyType> om;
        om.divide().divide(0);
        for (int j = 0; j < 8; ++j)
            om.divide(0, j);
        om.divide(0, 0, 0);

        // L3 in first octant, L1 otherwise
        leaves = om.makeTree();
        octree.resize(nNodes(leaves));
        ov = octree.data();
        updateInternalTree<KeyType>(leaves, ov);
    }

    std::vector<KeyType> leaves;
    OctreeData<KeyType, CpuTag> octree;
    OctreeView<KeyType> ov;
    std::vector<SourceCenterType<T>> centers;
    std::vector<uint8_t> macs;
};

TEST_F(MacRefinement, fullSurface)
{
    Box<T> box(0, 1);
    float invTheta = sqrt(3) / 2 + 1e-6;

    KeyType focusStart = 0;
    KeyType focusEnd   = decodePlaceholderBit(KeyType(011));
    while (!macRefine(octree, leaves, centers, macs, focusEnd, focusEnd, focusStart, focusEnd, invTheta, box)) {}

    int numNodesVertex = 7 + 8;
    int numNodesEdge   = 6 + 2 * 8;
    int numNodesFace   = 4 + 4 * 8;
    EXPECT_EQ(nNodes(leaves), 64 + 7 + 3 * numNodesFace + 3 * numNodesEdge + numNodesVertex);
}

TEST_F(MacRefinement, noSurface)
{
    Box<T> box(0, 1);
    float invTheta              = sqrt(3) / 2 + 1e-6;
    TreeNodeIndex numNodesStart = octree.numLeafNodes;

    KeyType oldFStart  = decodePlaceholderBit(KeyType(0101));
    KeyType oldFEnd    = decodePlaceholderBit(KeyType(011));
    KeyType focusStart = 0;
    KeyType focusEnd   = decodePlaceholderBit(KeyType(011));
    while (!macRefine(octree, leaves, centers, macs, oldFStart, oldFEnd, focusStart, focusEnd, invTheta, box)) {}

    EXPECT_EQ(nNodes(leaves), numNodesStart);
}

TEST_F(MacRefinement, partialSurface)
{
    Box<T> box(0, 1);
    float invTheta              = sqrt(3) / 2 + 1e-6;
    TreeNodeIndex numNodesStart = octree.numLeafNodes;

    KeyType oldFStart  = 0;
    KeyType oldFEnd    = decodePlaceholderBit(KeyType(0107));
    KeyType focusStart = 0;
    KeyType focusEnd   = decodePlaceholderBit(KeyType(011));
    while (!macRefine(octree, leaves, centers, macs, oldFStart, oldFEnd, focusStart, focusEnd, invTheta, box)) {}

    EXPECT_EQ(nNodes(leaves), numNodesStart + 5 * 7);
}

} // namespace cstone
