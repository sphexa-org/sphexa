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
void checkConnectivity(const Octree<KeyType>& fullTree)
{
    // check all internal nodes
    for (TreeNodeIndex nodeIdx = 0; nodeIdx < fullTree.numTreeNodes(); ++nodeIdx)
    {
        if (!fullTree.isLeaf(nodeIdx))
        {
            KeyType prefix = fullTree.codeStart(nodeIdx);
            int level      = fullTree.level(nodeIdx);

            EXPECT_EQ(fullTree.codeEnd(nodeIdx), prefix + nodeRange<KeyType>(level));

            for (int octant = 0; octant < 8; ++octant)
            {
                TreeNodeIndex child = fullTree.child(nodeIdx, octant);
                EXPECT_EQ(prefix + octant * nodeRange<KeyType>(level + 1), fullTree.codeStart(child));
            }

            if (nodeIdx > 0)
            {
                TreeNodeIndex parent = fullTree.parent(nodeIdx);
                EXPECT_EQ(fullTree.level(parent), level - 1);

                KeyType parentPrefix = fullTree.codeStart(parent);
                EXPECT_EQ(parentPrefix, enclosingBoxCode(prefix, level - 1));
            }
            else
            {
                EXPECT_EQ(fullTree.codeStart(nodeIdx), 0);
                EXPECT_EQ(fullTree.level(nodeIdx), 0);
            }
        }
        else
        {
            KeyType prefix = fullTree.codeStart(nodeIdx);
            int level      = fullTree.level(nodeIdx);

            EXPECT_EQ(fullTree.codeEnd(nodeIdx), prefix + nodeRange<KeyType>(level));

            TreeNodeIndex parent = fullTree.parent(nodeIdx);
            EXPECT_EQ(fullTree.level(parent), level - 1);

            KeyType parentPrefix = fullTree.codeStart(parent);
            EXPECT_EQ(parentPrefix, enclosingBoxCode(prefix, level - 1));
        }
    }
}

TEST(InternalOctree, rootNode)
{
    auto tree = makeRootNodeTree<unsigned>();

    Octree<unsigned> fullTree;
    fullTree.update(tree.data(), nNodes(tree));

    EXPECT_EQ(fullTree.numLeafNodes(), 1);
    EXPECT_EQ(fullTree.numTreeNodes(), 1);
    EXPECT_EQ(fullTree.numInternalNodes(), 0);
    EXPECT_EQ(fullTree.codeStart(0), 0);
    EXPECT_EQ(fullTree.codeEnd(0), nodeRange<unsigned>(0));
    EXPECT_EQ(fullTree.parent(0), 0);
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

    Octree<KeyType> fullTree;
    fullTree.update(tree.data(), nNodes(tree));

    ASSERT_EQ(fullTree.numInternalNodes(), (64 - 1) / 7);
    ASSERT_EQ(fullTree.numLeafNodes(), 64);

    EXPECT_EQ(fullTree.numTreeNodes(0), 1);
    EXPECT_EQ(fullTree.numTreeNodes(1), 8);
    EXPECT_EQ(fullTree.numTreeNodes(2), 64);
    EXPECT_EQ(fullTree.levelRange().back(), 73);

    EXPECT_EQ(fullTree.codeEnd(fullTree.toInternal(nNodes(tree) - 1)), nodeRange<KeyType>(0));

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

    Octree<KeyType> fullTree;
    fullTree.update(tree.data(), nNodes(tree));

    ASSERT_EQ(fullTree.numInternalNodes(), (15 - 1) / 7);
    ASSERT_EQ(fullTree.numLeafNodes(), 15);

    EXPECT_EQ(fullTree.numTreeNodes(0), 1);
    EXPECT_EQ(fullTree.numTreeNodes(1), 8);
    EXPECT_EQ(fullTree.numTreeNodes(2), 8);

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

    Octree<KeyType> fullTree;
    fullTree.update(tree.data(), nNodes(tree));
    EXPECT_EQ(fullTree.numTreeNodes(), 33);
    EXPECT_EQ(fullTree.numLeafNodes(), 29);
    EXPECT_EQ(fullTree.numInternalNodes(), 4);

    EXPECT_EQ(fullTree.numTreeNodes(0), 1);
    EXPECT_EQ(fullTree.numTreeNodes(1), 8);
    EXPECT_EQ(fullTree.numTreeNodes(2), 16);
    EXPECT_EQ(fullTree.numTreeNodes(3), 8);

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

    Octree<KeyType> fullTree;
    fullTree.update(spanningTree.data(), nNodes(spanningTree));

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

        Octree<KeyType> fullTree;
        fullTree.update(spanningTree.data(), nNodes(spanningTree));

        for (TreeNodeIndex i = 0; i < fullTree.numTreeNodes(); ++i)
        {
            KeyType key1 = fullTree.codeStart(i);
            KeyType key2 = fullTree.codeEnd(i);

            EXPECT_EQ(i, locateNode(key1, key2, fullTree.nodeKeys().data(), fullTree.levelRange().data()));
        }
    }
    {
        std::vector<KeyType> tree = makeUniformNLevelTree<KeyType>(4096, 1);
        Octree<KeyType> fullTree;
        fullTree.update(tree.data(), nNodes(tree));

        for (TreeNodeIndex i = 0; i < fullTree.numTreeNodes(); ++i)
        {
            KeyType key1 = fullTree.codeStart(i);
            KeyType key2 = fullTree.codeEnd(i);

            EXPECT_EQ(i, locateNode(key1, key2, fullTree.nodeKeys().data(), fullTree.levelRange().data()));
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

    OctreeData<KeyType, CpuTag> tree;
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
static void cstoneIndex()
{
    std::vector<KeyType> cornerstones{0, 1, nodeRange<KeyType>(0) - 1, nodeRange<KeyType>(0)};
    std::vector<KeyType> spanningTree = computeSpanningTree<KeyType>(cornerstones);

    Octree<KeyType> fullTree;
    fullTree.update(spanningTree.data(), nNodes(spanningTree));

    for (TreeNodeIndex i = 0; i < fullTree.numTreeNodes(); ++i)
    {
        if (fullTree.isLeaf(i))
        {
            KeyType internalKey = fullTree.codeStart(i);

            TreeNodeIndex cstoneIndex = fullTree.cstoneIndex(i);
            EXPECT_EQ(internalKey, spanningTree[cstoneIndex]);
        }
    }
}

TEST(InternalOctree, cstoneIndex)
{
    cstoneIndex<unsigned>();
    cstoneIndex<uint64_t>();
}

template<class KeyType>
static void upsweepSumIrregularL3()
{
    std::vector<KeyType> cstoneTree = OctreeMaker<KeyType>{}.divide().divide(0).divide(0, 2).divide(3).makeTree();
    Octree<KeyType> octree;
    octree.update(cstoneTree.data(), nNodes(cstoneTree));

    std::vector<unsigned> leafCounts(nNodes(cstoneTree), 1);
    std::vector<unsigned> nodeCounts(octree.numTreeNodes());

    scatter(octree.internalOrder(), leafCounts.data(), nodeCounts.data());
    upsweep(octree.levelRange(), octree.childOffsets().data(), nodeCounts.data(), NodeCount<unsigned>{});

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
