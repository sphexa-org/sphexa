/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief  Compute the internal part of a cornerstone octree on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/fill.h>

#include "cstone/primitives/math.hpp"
#include "cstone/sfc/common.hpp"
#include "cstone/tree/octree_gpu.h"

namespace cstone
{

/*! @brief combine internal and leaf tree parts into a single array with the nodeKey prefixes
 *
 * @tparam     KeyType           unsigned 32- or 64-bit integer
 * @param[in]  leaves            cornerstone SFC keys, length numLeafNodes + 1
 * @param[in]  numInternalNodes  number of internal octree nodes
 * @param[in]  numLeafNodes      total number of nodes
 * @param[in]  binaryToOct       translation map from binary to octree nodes
 * @param[out] prefixes          output octree SFC keys, length @p numInternalNodes + numLeafNodes
 *                               NOTE: keys are prefixed with Warren-Salmon placeholder bits!
 * @param[out] nodeOrder         iota 0,1,2,3,... sequence for later use, length same as @p prefixes
 */
template<class KeyType>
__global__ void createUnsortedLayout(const KeyType* leaves,
                                     TreeNodeIndex numInternalNodes,
                                     TreeNodeIndex numLeafNodes,
                                     KeyType* prefixes,
                                     TreeNodeIndex* nodeOrder)
{
    int tid = int(blockDim.x * blockIdx.x + threadIdx.x);
    if (tid < numLeafNodes)
    {
        KeyType key                       = leaves[tid];
        unsigned level                    = treeLevel(leaves[tid + 1] - key);
        prefixes[tid + numInternalNodes]  = encodePlaceholderBit(key, 3 * level);
        nodeOrder[tid + numInternalNodes] = tid + numInternalNodes;

        unsigned prefixLength = commonPrefix(key, leaves[tid + 1]);
        if (prefixLength % 3 == 0 && tid < numLeafNodes - 1)
        {
            TreeNodeIndex octIndex = (tid + binaryKeyWeight(key, prefixLength / 3)) / 7;
            prefixes[octIndex]     = encodePlaceholderBit(key, prefixLength);
            nodeOrder[octIndex]    = octIndex;
        }
    }
}

/*! @brief extract parent/child relationships from binary tree and translate to sorted order
 *
 * @tparam     KeyType           unsigned 32- or 64-bit integer
 * @param[in]  prefixes          octree node prefixes in Warren-Salmon format
 * @param[in]  numInternalNodes  number of internal octree nodes
 * @param[in]  leafToInternal    translation map from unsorted layout to level/SFC sorted octree layout
 *                               length is total number of octree nodes, internal + leaves
 * @param[in]  levelRange        indices of the first node at each level
 * @param[out] childOffsets      octree node index of first child for each node, length is total number of nodes
 * @param[out] parents           parent index of for each node which is the first of 8 siblings
 *                               i.e. the parent of node i is stored at parents[(i - 1)/8]
 */
template<class KeyType>
__global__ void linkTree(const KeyType* prefixes,
                         TreeNodeIndex numInternalNodes,
                         const TreeNodeIndex* leafToInternal,
                         const TreeNodeIndex* levelRange,
                         TreeNodeIndex* childOffsets,
                         TreeNodeIndex* parents)
{
    // loop over octree nodes index in unsorted layout A
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numInternalNodes)
    {
        TreeNodeIndex idxA    = leafToInternal[tid];
        KeyType prefix        = prefixes[idxA];
        KeyType nodeKey       = decodePlaceholderBit(prefix);
        unsigned prefixLength = decodePrefixLength(prefix);
        unsigned level        = prefixLength / 3;
        assert(level < maxTreeLevel<KeyType>{});

        KeyType childPrefix = encodePlaceholderBit(nodeKey, prefixLength + 3);

        TreeNodeIndex leafSearchStart = levelRange[level + 1];
        TreeNodeIndex leafSearchEnd   = levelRange[level + 2];
        TreeNodeIndex childIdx =
            stl::lower_bound(prefixes + leafSearchStart, prefixes + leafSearchEnd, childPrefix) - prefixes;

        if (childIdx != leafSearchEnd && childPrefix == prefixes[childIdx])
        {
            childOffsets[idxA] = childIdx;
            // We only store the parent once for every group of 8 siblings.
            // This works as long as each node always has 8 siblings.
            // Subtract one because the root has no siblings.
            parents[(childIdx - 1) / 8] = idxA;
        }
    }
}

//! @brief determine the octree subdivision level boundaries
template<class KeyType>
__global__ void getLevelRange(const KeyType* nodeKeys, TreeNodeIndex numNodes, TreeNodeIndex* levelRange)
{
    unsigned level    = blockIdx.x;
    auto it           = stl::lower_bound(nodeKeys, nodeKeys + numNodes, encodePlaceholderBit(KeyType(0), 3 * level));
    levelRange[level] = TreeNodeIndex(it - nodeKeys);

    if (level == maxTreeLevel<KeyType>{} + 1) { levelRange[level] = numNodes; }
}

//! @brief computes the inverse of the permutation given by @p order and then subtract @p numInternalNodes from it
__global__ void
invertOrder(TreeNodeIndex* order, TreeNodeIndex* inverseOrder, TreeNodeIndex numNodes, TreeNodeIndex numInternalNodes)
{
    int tid = int(blockDim.x * blockIdx.x + threadIdx.x);
    if (tid < numNodes)
    {
        inverseOrder[order[tid]] = tid;
        order[tid] -= numInternalNodes;
    }
}

template<class KeyType>
void buildOctreeGpu(const KeyType* cstoneTree, OctreeView<KeyType> d)
{
    constexpr unsigned numThreads = 256;

    TreeNodeIndex numNodes = d.numInternalNodes + d.numLeafNodes;
    createUnsortedLayout<<<iceil(numNodes, numThreads), numThreads>>>(cstoneTree, d.numInternalNodes, d.numLeafNodes,
                                                                      d.prefixes, d.internalToLeaf);

    thrust::sort_by_key(thrust::device, d.prefixes, d.prefixes + numNodes, d.internalToLeaf);

    invertOrder<<<iceil(numNodes, numThreads), numThreads>>>(d.internalToLeaf, d.leafToInternal, numNodes,
                                                             d.numInternalNodes);
    getLevelRange<<<maxTreeLevel<KeyType>{} + 2, 1>>>(d.prefixes, numNodes, d.levelRange);

    thrust::fill(thrust::device, d.childOffsets, d.childOffsets + numNodes, 0);
    if (d.numInternalNodes)
    {
        linkTree<<<iceil(d.numInternalNodes, numThreads), numThreads>>>(
            d.prefixes, d.numInternalNodes, d.leafToInternal, d.levelRange, d.childOffsets, d.parents);
    }
}

template void buildOctreeGpu(const uint32_t*, OctreeView<uint32_t>);
template void buildOctreeGpu(const uint64_t*, OctreeView<uint64_t>);

__global__ void upsweepSumKernel(TreeNodeIndex firstCell,
                                 TreeNodeIndex lastCell,
                                 const TreeNodeIndex* childOffsets,
                                 LocalIndex* nodeCounts)
{
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x + firstCell;
    if (cellIdx >= lastCell) return;

    TreeNodeIndex firstChild = childOffsets[cellIdx];

    if (firstChild) { nodeCounts[cellIdx] = NodeCount<LocalIndex>{}(cellIdx, firstChild, nodeCounts); }
}

void upsweepSumGpu(int numLevels,
                   const TreeNodeIndex* levelRange,
                   const TreeNodeIndex* childOffsets,
                   LocalIndex* nodeCounts)
{
    constexpr int numThreads = 128;

    for (int level = numLevels - 1; level >= 0; level--)
    {
        int numCellsLevel = levelRange[level + 1] - levelRange[level];
        int numBlocks     = (numCellsLevel - 1) / numThreads + 1;
        if (numCellsLevel)
        {
            upsweepSumKernel<<<numBlocks, numThreads>>>(levelRange[level], levelRange[level + 1], childOffsets,
                                                        nodeCounts);
        }
    }
}

} // namespace cstone
