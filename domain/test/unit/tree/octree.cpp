/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief octree utility tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * This file implements tests for OctreeMaker.
 * OctreeMaker can be used to generate octrees in cornerstone
 * format. It is only used to test the octree implementation.
 */

#include "gtest/gtest.h"

#include "cstone/tree/octree.hpp"
#include "cstone/tree/cs_util.hpp"

using namespace cstone;

template<class KeyType>
void checkConnectivity(OctreeData<KeyType, execution::Cpu>& fullTree)
{
    auto prefixes     = fullTree.prefixes.data();
    auto childOffsets = fullTree.childOffsets.data();
    auto parents      = fullTree.parents.data();

    // check all internal nodes
    for (TreeNodeIndex nodeIdx = 0; nodeIdx < fullTree.numNodes; ++nodeIdx)
    {
        KeyType prefix = decodePlaceholderBit(prefixes[nodeIdx]);
        unsigned level = decodePrefixLength(prefixes[nodeIdx]) / 3;

        if (childOffsets[nodeIdx] != 0)
        {
            for (int octant = 0; octant < 8; ++octant)
            {
                TreeNodeIndex child = childOffsets[nodeIdx] + octant;
                EXPECT_EQ(prefix + octant * nodeRange<KeyType>(level + 1), decodePlaceholderBit(prefixes[child]));
            }

            if (nodeIdx > 0)
            {
                TreeNodeIndex parent = parents[(nodeIdx - 1) / 8];
                EXPECT_EQ(decodePrefixLength(prefixes[parent]) / 3, level - 1);

                KeyType parentPrefix = decodePlaceholderBit(prefixes[parent]);
                EXPECT_EQ(parentPrefix, enclosingBoxCode(prefix, level - 1));
            }
            else
            {
                EXPECT_EQ(prefix, 0);
                EXPECT_EQ(level, 0u);
            }
        }
        else
        {
            TreeNodeIndex parent = nodeIdx ? parents[(nodeIdx - 1) / 8] : 0;
            EXPECT_EQ(decodePrefixLength(prefixes[parent]) / 3, level - 1);

            KeyType parentPrefix = decodePlaceholderBit(prefixes[parent]);
            EXPECT_EQ(parentPrefix, enclosingBoxCode(prefix, level - 1));
        }
    }
}

TEST(InternalOctree, rootNode)
{
    auto tree = makeRootNodeTree<unsigned>();

    OctreeData<unsigned, execution::Cpu> fullTree;
    fullTree.resize(nNodes(tree));
    updateInternalTree<unsigned>(tree, fullTree.data());

    EXPECT_EQ(fullTree.numLeafNodes, 1);
    EXPECT_EQ(fullTree.numNodes, 1);
    EXPECT_EQ(fullTree.numInternalNodes, 0);
    EXPECT_EQ(decodePlaceholderBit(fullTree.prefixes[0]), 0u);
    EXPECT_EQ(decodePlaceholderBit(fullTree.prefixes[0]) + nodeRange<unsigned>(0), nodeRange<unsigned>(0));
    EXPECT_EQ(0, 0); // parent of root is 0
}

/*! @brief test internal octree creation from a regular 4x4x4 grid of leaves
 *
 * This creates 64 level-2 leaf nodes. The resulting internal tree should
 * have 9 nodes, the root node and the 8 level-1 nodes.
 * The children of the root point to the level-1 nodes while the children
 * of the level-1 nodes point to the leaf nodes, i.e. the tree provided for constructing,
 * which is a separate array.
 */
template<class KeyType>
static void octree4x4x4()
{
    std::vector<KeyType> tree = makeUniformNLevelTree<KeyType>(64, 1);

    OctreeData<KeyType, execution::Cpu> fullTree;
    fullTree.resize(nNodes(tree));
    updateInternalTree<KeyType>(tree, fullTree.data());

    ASSERT_EQ(fullTree.numInternalNodes, (64 - 1) / 7);
    ASSERT_EQ(fullTree.numLeafNodes, 64);

    EXPECT_EQ(fullTree.levelRange[1] - fullTree.levelRange[0], 1);
    EXPECT_EQ(fullTree.levelRange[2] - fullTree.levelRange[1], 8);
    EXPECT_EQ(fullTree.levelRange[3] - fullTree.levelRange[2], 64);
    EXPECT_EQ(fullTree.levelRange.back(), 73);

    TreeNodeIndex lastLeafInternal = fullTree.leafToInternal[nNodes(tree) - 1 + fullTree.numInternalNodes];
    auto [rangeStart, rangeEnd]    = decodePlaceholderBit2K(fullTree.prefixes[lastLeafInternal]);
    EXPECT_EQ(rangeEnd, nodeRange<KeyType>(0));

    checkConnectivity<KeyType>(fullTree);
}

TEST(InternalOctree, octree4x4x4)
{
    octree4x4x4<unsigned>();
    octree4x4x4<uint64_t>();
}

/*! @brief test internal octree creation with an irregular leaf tree
 *
 * The leaf tree is the result of subdividing the root node, then further
 * subdividing octant 0. This results in 15 leaves, so the internal tree
 * should have two nodes: the root and the one internal level-1 node for the
 * first octant. The root points to the one internal node and to leaves [8:15].
 * The internal level-1 nodes points to leaves [0:8].
 */
template<class KeyType>
static void octreeIrregularL2()
{
    std::vector<KeyType> tree = OctreeMaker<KeyType>{}.divide().divide(0).makeTree();

    OctreeData<KeyType, execution::Cpu> fullTree;
    fullTree.resize(nNodes(tree));
    updateInternalTree<KeyType>(tree, fullTree.data());

    ASSERT_EQ(fullTree.numInternalNodes, (15 - 1) / 7);
    ASSERT_EQ(fullTree.numLeafNodes, 15);

    EXPECT_EQ(fullTree.levelRange[1] - fullTree.levelRange[0], 1);
    EXPECT_EQ(fullTree.levelRange[2] - fullTree.levelRange[1], 8);
    EXPECT_EQ(fullTree.levelRange[3] - fullTree.levelRange[2], 8);

    checkConnectivity<KeyType>(fullTree);
}

TEST(InternalOctree, irregularL2)
{
    octreeIrregularL2<unsigned>();
    octreeIrregularL2<uint64_t>();
}

//! @brief This creates an irregular tree. Checks geometry relations between children and parents.
template<class KeyType>
static void octreeIrregularL3()
{
    std::vector<KeyType> tree = OctreeMaker<KeyType>{}.divide().divide(0).divide(0, 2).divide(3).makeTree();

    OctreeData<KeyType, execution::Cpu> fullTree;
    fullTree.resize(nNodes(tree));
    updateInternalTree<KeyType>(tree, fullTree.data());
    EXPECT_EQ(fullTree.numNodes, 33);
    EXPECT_EQ(fullTree.numLeafNodes, 29);
    EXPECT_EQ(fullTree.numInternalNodes, 4);

    EXPECT_EQ(fullTree.levelRange[1] - fullTree.levelRange[0], 1);
    EXPECT_EQ(fullTree.levelRange[2] - fullTree.levelRange[1], 8);
    EXPECT_EQ(fullTree.levelRange[3] - fullTree.levelRange[2], 16);
    EXPECT_EQ(fullTree.levelRange[4] - fullTree.levelRange[3], 8);

    checkConnectivity<KeyType>(fullTree);
}

TEST(InternalOctree, irregularL3)
{
    octreeIrregularL3<unsigned>();
    octreeIrregularL3<uint64_t>();
}

//! @brief this generates a max-depth cornerstone tree
template<class KeyType>
static void spanningTree()
{
    std::vector<KeyType> cornerstones{0, 1, 030173, 03333333333, nodeRange<KeyType>(0) - 1, nodeRange<KeyType>(0)};
    std::vector<KeyType> spanningTree = computeSpanningTree<KeyType>(cornerstones);

    OctreeData<KeyType, execution::Cpu> fullTree;
    fullTree.resize(nNodes(spanningTree));
    updateInternalTree<KeyType>(spanningTree, fullTree.data());

    checkConnectivity(fullTree);
}

TEST(InternalOctree, spanningTree)
{
    spanningTree<unsigned>();
    spanningTree<uint64_t>();
}

template<class KeyType>
static void binaryIndexConversion()
{
    // a non-trivial tree that goes down to the maximum tree level in three different areas
    std::vector<KeyType> cornerstones{0, 1, 030173, 03333333333, nodeRange<KeyType>(0) - 1, nodeRange<KeyType>(0)};
    std::vector<KeyType> cstree = computeSpanningTree<KeyType>(cornerstones);

    TreeNodeIndex numNodes = nNodes(cstree);
    std::vector<TreeNodeIndex> octreeIndices(numNodes);
    for (TreeNodeIndex tid = 0; tid < numNodes; ++tid)
    {
        int prefixLength   = commonPrefix(cstree[tid], cstree[tid + 1]);
        bool divisibleBy3  = prefixLength % 3 == 0;
        octreeIndices[tid] = (divisibleBy3) ? 1 : 0;
    }
    std::vector<TreeNodeIndex> binaryToOct(numNodes);
    std::exclusive_scan(begin(octreeIndices), end(octreeIndices), begin(binaryToOct), 0);

    for (TreeNodeIndex tid = 0; tid < numNodes; ++tid)
    {
        int prefixLength  = commonPrefix(cstree[tid], cstree[tid + 1]);
        bool divisibleBy3 = prefixLength % 3 == 0;
        if (divisibleBy3)
        {
            TreeNodeIndex octIndex = (tid + binaryKeyWeight(cstree[tid], prefixLength / 3)) / 7;
            // The binaryKeyWeight formula yields the same result as an enumeration of the by-3 divisible
            // nodes, followed by a scan.
            EXPECT_EQ(octIndex, binaryToOct[tid]);
        }
    }
}

TEST(InternalOctree, binaryIndexConversion)
{
    binaryIndexConversion<unsigned>();
    binaryIndexConversion<uint64_t>();
}

template<class KeyType>
static void locate()
{
    {
        std::vector<KeyType> cornerstones{0, 1, nodeRange<KeyType>(0) - 1, nodeRange<KeyType>(0)};
        std::vector<KeyType> spanningTree = computeSpanningTree<KeyType>(cornerstones);

        OctreeData<KeyType, execution::Cpu> fullTree;
        fullTree.resize(nNodes(spanningTree));
        updateInternalTree<KeyType>(spanningTree, fullTree.data());

        for (TreeNodeIndex i = 0; i < fullTree.numNodes; ++i)
        {
            auto [key1, key2] = decodePlaceholderBit2K(fullTree.prefixes[i]);

            EXPECT_EQ(i, locateNode(key1, key2, fullTree.prefixes.data(), fullTree.levelRange.data()));
        }
    }
    {
        std::vector<KeyType> tree = makeUniformNLevelTree<KeyType>(4096, 1);
        OctreeData<KeyType, execution::Cpu> fullTree;
        fullTree.resize(nNodes(tree));
        updateInternalTree<KeyType>(tree, fullTree.data());

        for (TreeNodeIndex i = 0; i < fullTree.numNodes; ++i)
        {
            auto [key1, key2] = decodePlaceholderBit2K(fullTree.prefixes[i]);

            EXPECT_EQ(i, locateNode(key1, key2, fullTree.prefixes.data(), fullTree.levelRange.data()));
        }
    }
}

TEST(InternalOctree, locate)
{
    locate<unsigned>();
    locate<uint64_t>();
}

template<class KeyType>
static void containingNodeTrav()
{
    std::vector<KeyType> cornerstones{0, 1, nodeRange<KeyType>(0) - 1, nodeRange<KeyType>(0)};
    std::vector<KeyType> spanningTree = computeSpanningTree<KeyType>(cornerstones);

    OctreeData<KeyType, execution::Cpu> tree;
    tree.resize(nNodes(spanningTree));
    updateInternalTree<KeyType>(spanningTree, tree.data());

    for (TreeNodeIndex i = 0; i < tree.numNodes; ++i)
    {
        EXPECT_EQ(i, containingNode(tree.prefixes[i], tree.prefixes.data(), tree.childOffsets.data()));
    }

    EXPECT_EQ(011, tree.prefixes[containingNode(KeyType(0110), tree.prefixes.data(), tree.childOffsets.data())]);
    EXPECT_EQ(012, tree.prefixes[containingNode(KeyType(01202374), tree.prefixes.data(), tree.childOffsets.data())]);
    EXPECT_EQ(01001, tree.prefixes[containingNode(KeyType(010017), tree.prefixes.data(), tree.childOffsets.data())]);
}

TEST(InternalOctree, containingNode)
{
    containingNodeTrav<unsigned>();
    containingNodeTrav<uint64_t>();
}

TEST(InternalOctree, maxDepth)
{
    {
        std::vector<TreeNodeIndex> levelOffsets{0, 1, 1};
        EXPECT_EQ(maxDepth(levelOffsets.data(), levelOffsets.size()), 0);
    }
    {
        std::vector<TreeNodeIndex> levelOffsets{0, 1, 9, 9, 9, 9, 9, 9, 9};
        EXPECT_EQ(maxDepth(levelOffsets.data(), levelOffsets.size()), 1);
    }
    {
        std::vector<TreeNodeIndex> levelOffsets{0, 1, 9, 64};
        EXPECT_EQ(maxDepth(levelOffsets.data(), levelOffsets.size()), 2);
    }
}


template<class KeyType>
static void upsweepSumIrregularL3()
{
    std::vector<KeyType> cstoneTree = OctreeMaker<KeyType>{}.divide().divide(0).divide(0, 2).divide(3).makeTree();
    OctreeData<KeyType, execution::Cpu> octree;
    octree.resize(nNodes(cstoneTree));
    updateInternalTree<KeyType>(cstoneTree, octree.data());

    std::vector<unsigned> leafCounts(nNodes(cstoneTree), 1);
    std::vector<unsigned> nodeCounts(octree.numNodes);

    scatter(std::span<const TreeNodeIndex>{octree.leafToInternal.data() + octree.numInternalNodes,
                                           size_t(octree.numLeafNodes)},
            leafCounts.data(), nodeCounts.data());
    upsweep(std::span<const TreeNodeIndex>{octree.levelRange.data(), maxTreeLevel<KeyType>{} + 2},
            octree.childOffsets.data(), nodeCounts.data(), NodeCount<unsigned>{});

    //                                      L1                       L2
    //                                                               00                       30
    std::vector<unsigned> refNodeCounts{29, 15, 1, 1, 8, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        //  L3
                                        // 020
                                        1, 1, 1, 1, 1, 1, 1, 1};

    EXPECT_EQ(nodeCounts, refNodeCounts);
    EXPECT_EQ(nodeCounts[0], 29);
}

TEST(Upsweep, sumIrregularL3)
{
    upsweepSumIrregularL3<unsigned>();
    upsweepSumIrregularL3<uint64_t>();
}
