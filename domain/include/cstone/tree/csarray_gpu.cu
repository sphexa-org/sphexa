/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Generation of local and global octrees in cornerstone format on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * See octree.hpp for a description of the cornerstone format.
 */

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "cstone/cuda/cuda_runtime.hpp"
#include "cstone/cuda/errorcheck.cuh"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/primitives/math.hpp"
#include "csarray.hpp"

#include "csarray_gpu.h"

namespace cstone
{

//! @brief see computeNodeCounts
template<class KeyType>
__global__ void computeNodeCountsKernel(const KeyType* tree,
                                        unsigned* counts,
                                        TreeNodeIndex nNodes,
                                        const KeyType* codesStart,
                                        const KeyType* codesEnd,
                                        unsigned maxCount)
{
    TreeNodeIndex tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < nNodes) { counts[tid] = calculateNodeCount(tree[tid], tree[tid + 1], codesStart, codesEnd, maxCount); }
}

//! @brief see updateNodeCounts
template<class KeyType>
__global__ void updateNodeCountsKernel(const KeyType* tree,
                                       unsigned* counts,
                                       TreeNodeIndex numNodes,
                                       const KeyType* codesStart,
                                       const KeyType* codesEnd,
                                       unsigned maxCount)
{
    TreeNodeIndex tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numNodes)
    {
        unsigned firstGuess     = counts[tid];
        TreeNodeIndex secondIdx = (tid + 1 < numNodes - 1) ? tid + 1 : numNodes - 1;
        unsigned secondGuess    = counts[secondIdx];

        counts[tid] = updateNodeCount(tid, tree, firstGuess, secondGuess, codesStart, codesEnd, maxCount);
    }
}

//! @brief used to communicate required node search range for computeNodeCountsKernel back to host
__device__ TreeNodeIndex populatedNodes[2];

template<class KeyType>
__global__ void
findPopulatedNodes(const KeyType* tree, TreeNodeIndex nNodes, const KeyType* codesStart, const KeyType* codesEnd)
{
    if (threadIdx.x == 0 && codesStart != codesEnd)
    {
        populatedNodes[0] = stl::upper_bound(tree, tree + nNodes, *codesStart) - tree - 1;
        populatedNodes[1] = stl::upper_bound(tree, tree + nNodes, *(codesEnd - 1)) - tree;
    }
    else
    {
        populatedNodes[0] = nNodes;
        populatedNodes[1] = nNodes;
    }
}

template<class KeyType>
void computeNodeCountsGpu(execution::Gpu exec,
                          const KeyType* tree,
                          unsigned* counts,
                          TreeNodeIndex numNodes,
                          std::span<const KeyType> keys,
                          unsigned maxCount,
                          bool useCountsAsGuess)
{
    TreeNodeIndex popNodes[2];

    findPopulatedNodes<<<1, 1, 0, exec>>>(tree, numNodes, keys.data(), keys.data() + keys.size());
    checkGpuErrors(cudaMemcpyFromSymbolAsync(popNodes, GPU_SYMBOL(populatedNodes), 2 * sizeof(TreeNodeIndex), 0,
                                             cudaMemcpyDeviceToHost, exec));
    checkGpuErrors(cudaStreamSynchronize(exec));

    checkGpuErrors(cudaMemsetAsync(counts, 0, popNodes[0] * sizeof(unsigned), exec));
    checkGpuErrors(cudaMemsetAsync(counts + popNodes[1], 0, (numNodes - popNodes[1]) * sizeof(unsigned), exec));

    if (popNodes[1] <= popNodes[0]) { return; }

    constexpr unsigned nThreads = 256;
    if (useCountsAsGuess)
    {
        thrust::exclusive_scan(thrustExecPolicy(exec), counts + popNodes[0], counts + popNodes[1],
                               counts + popNodes[0]);
        updateNodeCountsKernel<<<iceil(popNodes[1] - popNodes[0], nThreads), nThreads, 0, exec>>>(
            tree + popNodes[0], counts + popNodes[0], popNodes[1] - popNodes[0], keys.data(), keys.data() + keys.size(),
            maxCount);
    }
    else
    {
        computeNodeCountsKernel<<<iceil(popNodes[1] - popNodes[0], nThreads), nThreads, 0, exec>>>(
            tree + popNodes[0], counts + popNodes[0], popNodes[1] - popNodes[0], keys.data(), keys.data() + keys.size(),
            maxCount);
    }
}

template void computeNodeCountsGpu(
    execution::Gpu, const unsigned*, unsigned*, TreeNodeIndex, std::span<const unsigned>, unsigned, bool);
template void computeNodeCountsGpu(
    execution::Gpu, const uint64_t*, unsigned*, TreeNodeIndex, std::span<const uint64_t>, unsigned, bool);

//! @brief this symbol is used to keep track of octree structure changes and detect convergence
__device__ int rebalanceChangeCounter;

/*! @brief Compute split or fuse decision for each octree node in parallel
 *
 * @tparam KeyType         32- or 64-bit unsigned integer type
 * @param[in] tree         octree nodes given as Morton codes of length @a numNodes
 *                         needs to satisfy the octree invariants
 * @param[in] counts       output particle counts per node, length = @a numNodes
 * @param[in] numNodes     number of nodes in tree
 * @param[in] bucketSize   maximum particle count per (leaf) node and
 *                         minimum particle count (strictly >) for (implicit) internal nodes
 * @param[out] nodeOps     stores rebalance decision result for each node, length = @a numNodes
 * @param[out] converged   stores 0 upon return if converged, a non-zero positive integer otherwise.
 *                         The storage location is accessed concurrently and cuda-memcheck might detect
 *                         a data race, but this is irrelevant for correctness.
 *
 * For each node i in the tree, in nodeOps[i], stores
 *  - 0 if to be merged
 *  - 1 if unchanged,
 *  - 8 if to be split.
 */
template<class KeyType>
__global__ void rebalanceDecisionKernel(
    const KeyType* tree, const unsigned* counts, TreeNodeIndex numNodes, unsigned bucketSize, TreeNodeIndex* nodeOps)
{
    TreeNodeIndex tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numNodes)
    {
        int decision = calculateNodeOp(tree, tid, counts, bucketSize);
        if (decision != 1) { rebalanceChangeCounter = 1; }
        nodeOps[tid] = decision;
    }
}

/*! @brief construct new nodes in the balanced tree
 *
 * @tparam KeyType         32- or 64-bit unsigned integer type
 * @param[in]  oldTree     old cornerstone octree, length = numOldNodes + 1
 * @param[in]  nodeOps     transformation codes for old tree, length = numOldNodes + 1
 * @param[in]  numOldNodes number of nodes in @a oldTree
 * @param[out] newTree     the rebalanced tree, length = nodeOps[numOldNodes] + 1
 */
template<class KeyType>
__global__ void
processNodes(const KeyType* oldTree, const TreeNodeIndex* nodeOps, TreeNodeIndex numOldNodes, KeyType* newTree)
{
    TreeNodeIndex tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numOldNodes) { processNode(tid, oldTree, nodeOps, newTree); }
}

__global__ void resetRebalanceCounter() { rebalanceChangeCounter = 0; }

template<class KeyType>
TreeNodeIndex computeNodeOpsGpu(execution::Gpu exec,
                                const KeyType* tree,
                                TreeNodeIndex numNodes,
                                const unsigned* counts,
                                unsigned bucketSize,
                                TreeNodeIndex* nodeOps)
{
    resetRebalanceCounter<<<1, 1, 0, exec>>>();

    constexpr unsigned nThreads = 512;
    rebalanceDecisionKernel<<<iceil(numNodes, nThreads), nThreads, 0, exec>>>(tree, counts, numNodes, bucketSize,
                                                                              nodeOps);

    size_t nodeOpsSize = numNodes + 1;
    thrust::exclusive_scan(thrustExecPolicy(exec), nodeOps, nodeOps + nodeOpsSize, nodeOps);

    TreeNodeIndex newNumNodes;
    checkGpuErrors(
        cudaMemcpyAsync(&newNumNodes, nodeOps + nodeOpsSize - 1, sizeof(TreeNodeIndex), cudaMemcpyDeviceToHost, exec));
    checkGpuErrors(cudaStreamSynchronize(exec));

    return newNumNodes;
}

template TreeNodeIndex
computeNodeOpsGpu(execution::Gpu, const unsigned*, TreeNodeIndex, const unsigned*, unsigned, TreeNodeIndex*);
template TreeNodeIndex
computeNodeOpsGpu(execution::Gpu, const uint64_t*, TreeNodeIndex, const unsigned*, unsigned, TreeNodeIndex*);

template<class KeyType>
bool rebalanceTreeGpu(execution::Gpu exec,
                      const KeyType* tree,
                      TreeNodeIndex numNodes,
                      TreeNodeIndex newNumNodes,
                      const TreeNodeIndex* nodeOps,
                      KeyType* newTree)
{
    constexpr unsigned nThreads = 512;
    processNodes<<<iceil(numNodes, nThreads), nThreads, 0, exec>>>(tree, nodeOps, numNodes, newTree);
    thrust::fill_n(thrustExecPolicy(exec), thrust::device_pointer_cast(newTree + newNumNodes), 1,
                   nodeRange<KeyType>(0));

    int changeCounter;
    checkGpuErrors(cudaMemcpyFromSymbolAsync(&changeCounter, GPU_SYMBOL(rebalanceChangeCounter), sizeof(int), 0,
                                             cudaMemcpyDeviceToHost, exec));
    checkGpuErrors(cudaStreamSynchronize(exec));

    return changeCounter == 0;
}

template bool
rebalanceTreeGpu(execution::Gpu, const unsigned*, TreeNodeIndex, TreeNodeIndex, const TreeNodeIndex*, unsigned*);
template bool
rebalanceTreeGpu(execution::Gpu, const uint64_t*, TreeNodeIndex, TreeNodeIndex, const TreeNodeIndex*, uint64_t*);

template<class KeyType>
__global__ void countSfcGapsKernel(const KeyType* tree, TreeNodeIndex numNodes, TreeNodeIndex* nodeOps)
{
    TreeNodeIndex i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numNodes) { nodeOps[i] = spanSfcRange(tree[i], tree[i + 1]); }
}

template<class KeyType>
void countSfcGapsGpu(execution::Gpu exec, const KeyType* tree, TreeNodeIndex numNodes, TreeNodeIndex* nodeOps)
{
    constexpr unsigned nThreads = 512;
    countSfcGapsKernel<<<iceil(numNodes, nThreads), nThreads, 0, exec>>>(tree, numNodes, nodeOps);
}

template void countSfcGapsGpu(execution::Gpu, const uint32_t*, TreeNodeIndex, TreeNodeIndex*);
template void countSfcGapsGpu(execution::Gpu, const uint64_t*, TreeNodeIndex, TreeNodeIndex*);

template<class KeyType>
__global__ void
fillSfcGapsKernel(const KeyType* tree, TreeNodeIndex numNodes, const TreeNodeIndex* nodeOps, KeyType* newTree)
{
    TreeNodeIndex i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numNodes) { spanSfcRange(tree[i], tree[i + 1], newTree + nodeOps[i]); }
    if (i == numNodes) { newTree[nodeOps[i]] = tree[numNodes]; }
}

template<class KeyType>
void fillSfcGapsGpu(
    execution::Gpu exec, const KeyType* tree, TreeNodeIndex numNodes, const TreeNodeIndex* nodeOps, KeyType* newTree)
{
    constexpr unsigned nThreads = 128;
    fillSfcGapsKernel<<<iceil(numNodes + 1, nThreads), nThreads, 0, exec>>>(tree, numNodes, nodeOps, newTree);
}

template void fillSfcGapsGpu(execution::Gpu, const uint32_t*, TreeNodeIndex, const TreeNodeIndex*, uint32_t*);
template void fillSfcGapsGpu(execution::Gpu, const uint64_t*, TreeNodeIndex, const TreeNodeIndex*, uint64_t*);

} // namespace cstone
