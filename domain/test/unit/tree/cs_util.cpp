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

#include "cstone/sfc/morton.hpp"
#include "cstone/tree/csarray.hpp"
#include "cstone/tree/cs_util.hpp"

using namespace cstone;

//! @brief detect missing zero node
template<class KeyType>
void invariantHead()
{
    std::vector<KeyType> tree{pad(KeyType(01), 3), pad(KeyType(02), 3), pad(KeyType(03), 3), pad(KeyType(04), 3),
                              pad(KeyType(05), 3), pad(KeyType(06), 3), pad(KeyType(07), 3), nodeRange<KeyType>(0)};

    EXPECT_FALSE(checkOctreeInvariants(tree.data(), nNodes(tree)));
}

//! @brief detect missing end node
template<class KeyType>
void invariantTail()
{
    std::vector<KeyType> tree{
        pad(KeyType(00), 3), pad(KeyType(01), 3), pad(KeyType(02), 3), pad(KeyType(03), 3),
        pad(KeyType(04), 3), pad(KeyType(05), 3), pad(KeyType(06), 3), pad(KeyType(07), 3),
    };

    EXPECT_FALSE(checkOctreeInvariants(tree.data(), nNodes(tree)));
}

//! @brief detect missing siblings
template<class KeyType>
void invariantSiblings()
{
    std::vector<KeyType> tree{pad(KeyType(00), 3), pad(KeyType(01), 3), pad(KeyType(02), 3), pad(KeyType(03), 3),
                              pad(KeyType(04), 3), pad(KeyType(05), 3), pad(KeyType(06), 3), nodeRange<KeyType>(0)};

    EXPECT_FALSE(checkOctreeInvariants(tree.data(), nNodes(tree)));
}

TEST(CornerstoneUtil, invariants32)
{
    invariantHead<unsigned>();
    invariantTail<unsigned>();
    invariantSiblings<unsigned>();
}

TEST(CornerstoneUtil, invariants64)
{
    invariantHead<uint64_t>();
    invariantTail<uint64_t>();
    invariantSiblings<uint64_t>();
}

TEST(CornerstoneUtil, codeFromIndices32)
{
    using CodeType = unsigned;
    EXPECT_EQ(0x08000000, codeFromIndices<CodeType>({1}));
    EXPECT_EQ(0x09000000, codeFromIndices<CodeType>({1, 1}));
    EXPECT_EQ(0x09E00000, codeFromIndices<CodeType>({1, 1, 7}));
}

TEST(CornerstoneUtil, codeFromIndices64)
{
    using CodeType = uint64_t;
    EXPECT_EQ(0b0001lu << 60u, codeFromIndices<CodeType>({1}));
    EXPECT_EQ(0b0001001lu << 57u, codeFromIndices<CodeType>({1, 1}));
    EXPECT_EQ(0b0001001111lu << 54u, codeFromIndices<CodeType>({1, 1, 7}));
}

template<class KeyType>
void codeFromIndices()
{
    constexpr unsigned maxLevel = maxTreeLevel<KeyType>{};

    std::array<unsigned char, 21> input{0};
    for (unsigned i = 0; i < maxLevel; ++i)
    {
        input[i] = 7;
    }

    EXPECT_EQ(nodeRange<KeyType>(0), codeFromIndices<KeyType>(input) + 1);
}

TEST(CornerstoneUtil, codeFromIndices)
{
    codeFromIndices<unsigned>();
    codeFromIndices<uint64_t>();
}

//! @brief test OctreeMaker node division
template<class KeyType>
void octreeMakerDivide()
{
    using CodeType = KeyType;

    // divide root node and node {7}
    auto tree = OctreeMaker<CodeType>{}.divide().divide(7).makeTree();

    std::vector<CodeType> refTree{
        pad(CodeType(00), 3),  pad(CodeType(01), 3),  pad(CodeType(02), 3),  pad(CodeType(03), 3),
        pad(CodeType(04), 3),  pad(CodeType(05), 3),  pad(CodeType(06), 3),  pad(CodeType(070), 6),
        pad(CodeType(071), 6), pad(CodeType(072), 6), pad(CodeType(073), 6), pad(CodeType(074), 6),
        pad(CodeType(075), 6), pad(CodeType(076), 6), pad(CodeType(077), 6), nodeRange<CodeType>(0)};

    EXPECT_EQ(tree, refTree);
}

TEST(CornerstoneUtil, octreeMakerDivide32) { octreeMakerDivide<unsigned>(); }

TEST(CornerstoneUtil, octreeMakerDivide64) { octreeMakerDivide<uint64_t>(); }

//! @brief test OctreeMaker creation of a maximum level tree
template<class KeyType>
void octreeMakerMaxLevel()
{
    using CodeType = KeyType;

    std::vector<CodeType> refTree{0};
    {
        std::array<unsigned char, maxTreeLevel<uint64_t>{}> zeroIndices{0};
        for (std::size_t level = 0; level < maxTreeLevel<CodeType>{}; ++level)
        {
            auto indices = zeroIndices;
            for (std::size_t sibling = 1; sibling < 8; ++sibling)
            {
                indices[level] = sibling;
                refTree.push_back(codeFromIndices<CodeType>(indices));
            }
        }
        refTree.push_back(nodeRange<CodeType>(0));
        std::sort(begin(refTree), end(refTree));
    }

    EXPECT_TRUE(checkOctreeInvariants(refTree.data(), nNodes(refTree)));

    OctreeMaker<CodeType> octreeMaker;
    for (std::size_t level = 0; level < maxTreeLevel<KeyType>{}; ++level)
        octreeMaker.divide({}, level);

    std::vector<CodeType> tree = octreeMaker.makeTree();
    EXPECT_EQ(tree, refTree);
}

TEST(CornerstoneUtil, octreeMakerMaxLevel32) { octreeMakerMaxLevel<unsigned>(); }

TEST(CornerstoneUtil, octreeMakerMaxLevel64) { octreeMakerMaxLevel<uint64_t>(); }
