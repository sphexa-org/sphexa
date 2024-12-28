/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Binary radix tree creation tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/tree/btree.hpp"
#include "cstone/tree/csarray.hpp"
#include "cstone/tree/cs_util.hpp"

/*! @brief @file tests for binary tree generation
 */

using namespace cstone;

TEST(BinaryTree, loadStoreIndex)
{
    EXPECT_TRUE(storeLeafIndex(0) < 0);
    EXPECT_EQ(loadLeafIndex(storeLeafIndex(0)), 0);

    TreeNodeIndex maxIndex = (1ul << (8 * sizeof(TreeNodeIndex) - 1)) - 1;
    EXPECT_TRUE(storeLeafIndex(maxIndex) < 0);
    EXPECT_EQ(loadLeafIndex(storeLeafIndex(maxIndex)), maxIndex);
}

//! @brief check binary node prefixes
template<class KeyType>
void internal4x4x4PrefixTest()
{
    // a tree with 4 subdivisions along each dimension, 64 nodes
    std::vector<KeyType> tree = makeUniformNLevelTree<KeyType>(64, 1);

    std::vector<BinaryNode<KeyType>> internalTree(nNodes(tree));
    createBinaryTree(tree.data(), nNodes(tree), internalTree.data());

    EXPECT_EQ(decodePrefixLength(internalTree[0].prefix), 0);
    EXPECT_EQ(internalTree[0].prefix, 1);

    EXPECT_EQ(decodePrefixLength(internalTree[31].prefix), 1);
    EXPECT_EQ(internalTree[31].prefix, KeyType(0b10));
    EXPECT_EQ(decodePrefixLength(internalTree[32].prefix), 1);
    EXPECT_EQ(internalTree[32].prefix, KeyType(0b11));

    EXPECT_EQ(decodePrefixLength(internalTree[15].prefix), 2);
    EXPECT_EQ(internalTree[15].prefix, KeyType(0b100));
    EXPECT_EQ(decodePrefixLength(internalTree[16].prefix), 2);
    EXPECT_EQ(internalTree[16].prefix, KeyType(0b101));

    EXPECT_EQ(decodePrefixLength(internalTree[7].prefix), 3);
    EXPECT_EQ(internalTree[7].prefix, KeyType(0b1000));
    EXPECT_EQ(decodePrefixLength(internalTree[8].prefix), 3);
    EXPECT_EQ(internalTree[8].prefix, KeyType(0b1001));

    // second (useless) root node
    EXPECT_EQ(decodePrefixLength(internalTree[63].prefix), 0);
    EXPECT_EQ(internalTree[63].prefix, 1);
}

TEST(BinaryTree, internalTree4x4x4PrefixTest)
{
    internal4x4x4PrefixTest<unsigned>();
    internal4x4x4PrefixTest<uint64_t>();
}

/*! Create a set of irregular octree leaves which do not cover the whole space
 *
 * This example is illustrated in the original paper referenced in sfc/binarytree.hpp.
 * Please refer to the publication for a graphical illustration of the resulting
 * node connectivity.
 */

template<class KeyType>
std::vector<KeyType> makeExample()
{
    std::vector<KeyType> ret{
        pad(KeyType(0b00001), 5), pad(KeyType(0b00010), 5), pad(KeyType(0b00100), 5), pad(KeyType(0b00101), 5),
        pad(KeyType(0b10011), 5), pad(KeyType(0b11000), 5), pad(KeyType(0b11001), 5), pad(KeyType(0b11110), 5),
    };
    return ret;
}

template<class KeyType>
void findSplitTest()
{
    std::vector<KeyType> example = makeExample<KeyType>();

    {
        TreeNodeIndex split = findSplit(example.data(), 0, 7);
        EXPECT_EQ(split, 3);
    }
    {
        TreeNodeIndex split = findSplit(example.data(), 0, 3);
        EXPECT_EQ(split, 1);
    }
    {
        TreeNodeIndex split = findSplit(example.data(), 4, 7);
        EXPECT_EQ(split, 4);
    }
}

TEST(BinaryTree, findSplit)
{
    findSplitTest<unsigned>();
    findSplitTest<uint64_t>();
}

template<class KeyType>
void paperExampleTest()
{
    using CodeType = KeyType;

    std::vector<CodeType> example = makeExample<CodeType>();
    std::vector<BinaryNode<CodeType>> internalNodes(example.size() - 1);
    for (std::size_t i = 0; i < internalNodes.size(); ++i)
    {
        constructInternalNode(example.data(), example.size(), internalNodes.data(), i);
    }

    std::vector<TreeNodeIndex> refLeft{3, storeLeafIndex(0), storeLeafIndex(2), 1, storeLeafIndex(4),
                                       6, storeLeafIndex(5)};

    std::vector<TreeNodeIndex> refRight{4, storeLeafIndex(1), storeLeafIndex(3), 2,
                                        5, storeLeafIndex(7), storeLeafIndex(6)};

    std::vector<TreeNodeIndex> refPrefixLengths{0, 3, 4, 2, 1, 2, 4};

    using Node = BinaryNode<KeyType>;
    for (std::size_t idx = 0; idx < internalNodes.size(); ++idx)
    {
        EXPECT_EQ(internalNodes[idx].child[Node::left], refLeft[idx]);
        EXPECT_EQ(internalNodes[idx].child[Node::right], refRight[idx]);
        EXPECT_EQ(decodePrefixLength(internalNodes[idx].prefix), refPrefixLengths[idx]);
    }
}

TEST(BinaryTree, internalIrregular)
{
    paperExampleTest<unsigned>();
    paperExampleTest<uint64_t>();
}
