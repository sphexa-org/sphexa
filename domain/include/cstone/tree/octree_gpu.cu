/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
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
#include "cstone/primitives/primitives_gpu.h"
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
void buildOctreeGpu(const KeyType* cstoneTree,
                    OctreeView<KeyType> d,
                    std::span<KeyType> keyBuf,
                    std::span<TreeNodeIndex> valueBuf,
                    std::span<char> cubTmp)
{
    constexpr unsigned numThreads = 256;

    TreeNodeIndex numNodes = d.numInternalNodes + d.numLeafNodes;
    createUnsortedLayout<<<iceil(numNodes, numThreads), numThreads>>>(cstoneTree, d.numInternalNodes, d.numLeafNodes,
                                                                      d.prefixes, d.internalToLeaf);

    assert(keyBuf.size() == d.numNodes && valueBuf.size() == d.numNodes);
    sortByKeyGpu(d.prefixes, d.prefixes + numNodes, d.internalToLeaf, keyBuf.data(), valueBuf.data(), cubTmp.data(),
                 cubTmp.size());

    invertOrder<<<iceil(numNodes, numThreads), numThreads>>>(d.internalToLeaf, d.leafToInternal, numNodes,
                                                             d.numInternalNodes);
    getLevelRange<<<maxTreeLevel<KeyType>{} + 2, 1>>>(d.prefixes, numNodes, d.d_levelRange);
    memcpyD2H(d.d_levelRange, maxTreeLevel<KeyType>{} + 2, d.levelRange);

    thrust::fill(thrust::device, d.childOffsets, d.childOffsets + numNodes, 0);
    if (d.numInternalNodes)
    {
        linkTree<<<iceil(d.numInternalNodes, numThreads), numThreads>>>(
            d.prefixes, d.numInternalNodes, d.leafToInternal, d.d_levelRange, d.childOffsets, d.parents);
    }
}

template void
buildOctreeGpu(const uint32_t*, OctreeView<uint32_t>, std::span<uint32_t>, std::span<TreeNodeIndex>, std::span<char>);
template void
buildOctreeGpu(const uint64_t*, OctreeView<uint64_t>, std::span<uint64_t>, std::span<TreeNodeIndex>, std::span<char>);

template<class KeyType>
void buildOctreeGpu(const KeyType* cstoneTree, OctreeView<KeyType> d)
{
    KeyType* keyBuf;
    TreeNodeIndex* valueBuf;
    char* cubTmp;
    uint64_t tmpStorage = sortByKeyTempStorage<KeyType, TreeNodeIndex>(d.numNodes);
    checkGpuErrors(cudaMalloc(&keyBuf, sizeof(KeyType) * d.numNodes));
    checkGpuErrors(cudaMalloc(&valueBuf, sizeof(TreeNodeIndex) * d.numNodes));
    checkGpuErrors(cudaMalloc(&cubTmp, tmpStorage));

    buildOctreeGpu(cstoneTree, d, {keyBuf, size_t(d.numNodes)}, {valueBuf, size_t(d.numNodes)}, {cubTmp, tmpStorage});

    checkGpuErrors(cudaFree(keyBuf));
    checkGpuErrors(cudaFree(valueBuf));
    checkGpuErrors(cudaFree(cubTmp));
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

template<class KeyType>
__global__ void locateNodesKernel(const KeyType* k1,
                                  const KeyType* k2,
                                  const KeyType* nodeKeys,
                                  const TreeNodeIndex* lvlRange,
                                  TreeNodeIndex* indices)
{
    LocalIndex tid  = blockIdx.x * blockDim.x + threadIdx.x;
    TreeNodeIndex n = k2 - k1 - 1;
    if (tid < n) { indices[tid] = locateNode(k1[tid], k1[tid + 1], nodeKeys, lvlRange); }
}

template<class KeyType>
void locateNodesGpu(const KeyType* k1,
                    const KeyType* k2,
                    const KeyType* nodeKeys,
                    const TreeNodeIndex* lvlRange,
                    TreeNodeIndex* indices)
{
    int numThreads = 256;
    int numBlocks  = iceil(k2 - k1 - 1, numThreads);
    if (numBlocks == 0) { return; }
    locateNodesKernel<<<numBlocks, numThreads>>>(k1, k2, nodeKeys, lvlRange, indices);
}

template void locateNodesGpu(const uint32_t* k1,
                             const uint32_t* k2,
                             const uint32_t* nodeKeys,
                             const TreeNodeIndex* lvlRange,
                             TreeNodeIndex* indices);
template void locateNodesGpu(const uint64_t* k1,
                             const uint64_t* k2,
                             const uint64_t* nodeKeys,
                             const TreeNodeIndex* lvlRange,
                             TreeNodeIndex* indices);

template<class KeyType>
__global__ void locateNodesKernel(const KeyType* k1,
                                  const TreeNodeIndex* map,
                                  size_t n,
                                  const KeyType* nodeKeys,
                                  const TreeNodeIndex* lvlRange,
                                  TreeNodeIndex* indices)
{
    LocalIndex tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) { indices[tid] = locateNode(k1[map[tid]], nodeKeys, lvlRange); }
}

template<class KeyType>
void locateNodesGpu(const KeyType* k1,
                    const TreeNodeIndex* map,
                    size_t n,
                    const KeyType* nodeKeys,
                    const TreeNodeIndex* lvlRange,
                    TreeNodeIndex* indices)
{
    int numThreads = 256;
    int numBlocks  = iceil(n, numThreads);
    if (numBlocks == 0) { return; }
    locateNodesKernel<<<numBlocks, numThreads>>>(k1, map, n, nodeKeys, lvlRange, indices);
}

template void locateNodesGpu(const uint32_t* k1,
                             const TreeNodeIndex* map,
                             size_t n,
                             const uint32_t* nodeKeys,
                             const TreeNodeIndex* lvlRange,
                             TreeNodeIndex* indices);
template void locateNodesGpu(const uint64_t* k1,
                             const TreeNodeIndex* map,
                             size_t n,
                             const uint64_t* nodeKeys,
                             const TreeNodeIndex* lvlRange,
                             TreeNodeIndex* indices);

} // namespace cstone
