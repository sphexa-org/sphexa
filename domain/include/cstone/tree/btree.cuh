/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @brief @file parallel binary radix tree construction CUDA kernel
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/primitives/math.hpp"
#include "btree.hpp"

namespace cstone
{

//! @brief see createBinaryTree
template<class KeyType>
__global__ void createBinaryTreeKernel(const KeyType* cstree, TreeNodeIndex numNodes, BinaryNode<KeyType>* binaryTree)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numNodes) { constructInternalNode(cstree, numNodes + 1, binaryTree, tid); }
}

//! @brief convenience kernel wrapper
template<class KeyType>
void createBinaryTreeGpu(const KeyType* cstree, TreeNodeIndex numNodes, BinaryNode<KeyType>* binaryTree)
{
    constexpr int numThreads = 256;
    createBinaryTreeKernel<<<iceil(numNodes, numThreads), numThreads>>>(cstree, numNodes, binaryTree);
}

} // namespace cstone