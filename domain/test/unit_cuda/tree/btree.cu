/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief binary radix tree GPU testing
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "gtest/gtest.h"

#include "cstone/tree/btree.cuh"
#include "cstone/tree/cs_util.hpp"

using namespace cstone;

//! @brief check binary node prefixes
template<class I>
void internal4x4x4PrefixTest()
{
    // a tree with 4 subdivisions along each dimension, 64 nodes
    thrust::device_vector<I> tree = makeUniformNLevelTree<I>(64, 1);

    thrust::device_vector<BinaryNode<I>> d_internalTree(nNodes(tree));
    createBinaryTreeGpu(thrust::raw_pointer_cast(tree.data()), nNodes(tree),
                        thrust::raw_pointer_cast(d_internalTree.data()));

    thrust::host_vector<BinaryNode<I>> internalTree = d_internalTree;

    EXPECT_EQ(decodePrefixLength(internalTree[0].prefix), 0);
    EXPECT_EQ(internalTree[0].prefix, 1);

    EXPECT_EQ(decodePrefixLength(internalTree[31].prefix), 1);
    EXPECT_EQ(internalTree[31].prefix, I(0b10));
    EXPECT_EQ(decodePrefixLength(internalTree[32].prefix), 1);
    EXPECT_EQ(internalTree[32].prefix, I(0b11));

    EXPECT_EQ(decodePrefixLength(internalTree[15].prefix), 2);
    EXPECT_EQ(internalTree[15].prefix, I(0b100));
    EXPECT_EQ(decodePrefixLength(internalTree[16].prefix), 2);
    EXPECT_EQ(internalTree[16].prefix, I(0b101));

    EXPECT_EQ(decodePrefixLength(internalTree[7].prefix), 3);
    EXPECT_EQ(internalTree[7].prefix, I(0b1000));
    EXPECT_EQ(decodePrefixLength(internalTree[8].prefix), 3);
    EXPECT_EQ(internalTree[8].prefix, I(0b1001));

    // second (useless) root node
    EXPECT_EQ(decodePrefixLength(internalTree[63].prefix), 0);
    EXPECT_EQ(internalTree[63].prefix, 1);
}

TEST(BinaryTreeGpu, internalTree4x4x4PrefixTest)
{
    internal4x4x4PrefixTest<unsigned>();
    internal4x4x4PrefixTest<uint64_t>();
}
