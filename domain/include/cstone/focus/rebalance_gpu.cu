/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Focused octree rebalance on GPUs
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/cuda/cub.hpp"
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "cstone/cuda/errorcheck.cuh"
#include "cstone/focus/rebalance.hpp"
#include "cstone/focus/rebalance_gpu.h"
#include "cstone/primitives/math.hpp"

namespace cstone
{

template<class KeyType>
__global__ void rebalanceDecisionEssentialKernel(const KeyType* prefixes,
                                                 const TreeNodeIndex* childOffsets,
                                                 const TreeNodeIndex* parents,
                                                 const unsigned* counts,
                                                 const uint8_t* macs,
                                                 KeyType focusStart,
                                                 KeyType focusEnd,
                                                 unsigned bucketSize,
                                                 TreeNodeIndex* nodeOps,
                                                 TreeNodeIndex numNodes)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numNodes)
    {
        nodeOps[tid] =
            mergeCountAndMacOp(tid, prefixes, childOffsets, parents, counts, macs, focusStart, focusEnd, bucketSize);
    }
}

template<class KeyType>
void rebalanceDecisionEssentialGpu(const KeyType* prefixes,
                                   const TreeNodeIndex* childOffsets,
                                   const TreeNodeIndex* parents,
                                   const unsigned* counts,
                                   const uint8_t* macs,
                                   KeyType focusStart,
                                   KeyType focusEnd,
                                   unsigned bucketSize,
                                   TreeNodeIndex* nodeOps,
                                   TreeNodeIndex numNodes)
{
    constexpr unsigned numThreads = 256;
    rebalanceDecisionEssentialKernel<<<iceil(numNodes, numThreads), numThreads>>>(
        prefixes, childOffsets, parents, counts, macs, focusStart, focusEnd, bucketSize, nodeOps, numNodes);
}

#define REBA_DEC_ESS_GPU(KeyType)                                                                                      \
    template void rebalanceDecisionEssentialGpu(const KeyType* prefixes, const TreeNodeIndex* childOffsets,            \
                                                const TreeNodeIndex* parents, const unsigned* counts,                  \
                                                const uint8_t* macs, KeyType focusStart, KeyType focusEnd,             \
                                                unsigned bucketSize, TreeNodeIndex* nodeOps, TreeNodeIndex numNodes)
REBA_DEC_ESS_GPU(uint32_t);
REBA_DEC_ESS_GPU(uint64_t);

template<class KeyType>
__global__ void macRefineDecisionKernel(const KeyType* prefixes,
                                        const uint8_t* macs,
                                        const TreeNodeIndex* l2i,
                                        TreeNodeIndex numLeafNodes,
                                        int2 focus,
                                        TreeNodeIndex* nodeOps)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numLeafNodes) { return; }

    if (i < focus.x || i >= focus.y) { nodeOps[i] = macRefineOp(prefixes[l2i[i]], macs[l2i[i]]); }
    else { nodeOps[i] = 1; }
}

template<class KeyType>
void macRefineDecisionGpu(const KeyType* prefixes,
                          const uint8_t* macs,
                          const TreeNodeIndex* l2i,
                          TreeNodeIndex numLeafNodes,
                          TreeIndexPair focus,
                          TreeNodeIndex* nodeOps)
{
    constexpr unsigned numThreads = 256;
    macRefineDecisionKernel<<<iceil(numLeafNodes, numThreads), numThreads>>>(prefixes, macs, l2i, numLeafNodes,
                                                                             {focus.start(), focus.end()}, nodeOps);
}

#define MAC_REF_DEC_GPU(KeyType)                                                                                       \
    template void macRefineDecisionGpu(const KeyType* prefixes, const uint8_t* macs, const TreeNodeIndex* l2i,         \
                                       TreeNodeIndex numLeafNodes, TreeIndexPair focus, TreeNodeIndex* nodeOps)
MAC_REF_DEC_GPU(uint32_t);
MAC_REF_DEC_GPU(uint64_t);

__device__ int nodeOpSum;
__global__ void resetNodeOpSum() { nodeOpSum = 0; }

template<class KeyType>
__global__ void protectAncestorsKernel(const KeyType* prefixes,
                                       const TreeNodeIndex* parents,
                                       TreeNodeIndex* nodeOps,
                                       TreeNodeIndex numNodes)
{
    int nodeOp = 1;

    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numNodes)
    {
        nodeOp = nzAncestorOp(tid, prefixes, parents, nodeOps);
        // technically a race condition since nodeOps[tid] might be read by another thread,
        // but all possible outcomes are identical
        nodeOps[tid] = nodeOp;
    }

    typedef cub::BlockReduce<int, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    BlockReduce reduce(temp_storage);
    int blockMin = reduce.Reduce(int(nodeOp != 1), cub::Sum());
    __syncthreads();

    if (threadIdx.x == 0) { atomicAdd(&nodeOpSum, blockMin); }
}

template<class KeyType>
bool protectAncestorsGpu(const KeyType* prefixes,
                         const TreeNodeIndex* parents,
                         TreeNodeIndex* nodeOps,
                         TreeNodeIndex numNodes)
{
    resetNodeOpSum<<<1, 1>>>();

    constexpr unsigned numThreads = 256;
    protectAncestorsKernel<<<iceil(numNodes, numThreads), numThreads>>>(prefixes, parents, nodeOps, numNodes);

    int numNodesModify;
    checkGpuErrors(cudaMemcpyFromSymbol(&numNodesModify, GPU_SYMBOL(nodeOpSum), sizeof(int)));

    return numNodesModify == 0;
}

template bool protectAncestorsGpu(const uint32_t*, const TreeNodeIndex*, TreeNodeIndex*, TreeNodeIndex);
template bool protectAncestorsGpu(const uint64_t*, const TreeNodeIndex*, TreeNodeIndex*, TreeNodeIndex);

__device__ int enforceKeyStatus_device;
__global__ void resetEnforceKeyStatus() { enforceKeyStatus_device = static_cast<int>(ResolutionStatus::converged); }

template<class KeyType>
__global__ void enforceKeysKernel(const KeyType* forcedKeys,
                                  const KeyType* nodeKeys,
                                  const TreeNodeIndex* childOffsets,
                                  const TreeNodeIndex* parents,
                                  TreeNodeIndex* nodeOps)
{
    unsigned i       = blockIdx.x;
    auto statusBlock = enforceKeySingle(forcedKeys[i], nodeKeys, childOffsets, parents, nodeOps);
    atomicMax(&enforceKeyStatus_device, static_cast<int>(statusBlock));
}

template<class KeyType>
ResolutionStatus enforceKeysGpu(const KeyType* forcedKeys,
                                TreeNodeIndex numForcedKeys,
                                const KeyType* nodeKeys,
                                const TreeNodeIndex* childOffsets,
                                const TreeNodeIndex* parents,
                                TreeNodeIndex* nodeOps)
{
    resetEnforceKeyStatus<<<1, 1>>>();
    if (numForcedKeys)
    {
        enforceKeysKernel<<<numForcedKeys, 1>>>(forcedKeys, nodeKeys, childOffsets, parents, nodeOps);
    }

    int status;
    checkGpuErrors(cudaMemcpyFromSymbol(&status, GPU_SYMBOL(enforceKeyStatus_device), sizeof(ResolutionStatus)));
    return static_cast<ResolutionStatus>(status);
}

#define ENFORCE_KEYS_GPU(KeyType)                                                                                      \
    template ResolutionStatus enforceKeysGpu(const KeyType* forcedKeys, TreeNodeIndex numForcedKeys,                   \
                                             const KeyType* nodeKeys, const TreeNodeIndex* childOffsets,               \
                                             const TreeNodeIndex* parents, TreeNodeIndex* nodeOps)
ENFORCE_KEYS_GPU(uint32_t);
ENFORCE_KEYS_GPU(uint64_t);

template<class KeyType>
__global__ void rangeCountKernel(const KeyType* leaves,
                                 TreeNodeIndex numLeaves,
                                 const unsigned* counts,
                                 const KeyType* leavesFocus,
                                 const TreeNodeIndex* leavesFocusIdx,
                                 TreeNodeIndex leavesFocusIdxSize,
                                 unsigned* countsFocus)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= leavesFocusIdxSize) { return; }

    TreeNodeIndex leafIdx = leavesFocusIdx[i];
    KeyType startKey      = leavesFocus[leafIdx];
    KeyType endKey        = leavesFocus[leafIdx + 1];

    size_t startIdx = findNodeBelow(leaves, numLeaves, startKey);
    size_t endIdx   = findNodeAbove(leaves, numLeaves, endKey);

    uint64_t globCount   = thrust::reduce(thrust::seq, counts + startIdx, counts + endIdx, uint64_t(0));
    countsFocus[leafIdx] = stl::min(uint64_t(0xFFFFFFFF), globCount);
}

template<class KeyType>
void rangeCountGpu(std::span<const KeyType> leaves,
                   std::span<const unsigned> counts,
                   std::span<const KeyType> leavesFocus,
                   std::span<const TreeNodeIndex> leavesFocusIdx,
                   std::span<unsigned> countsFocus)
{
    constexpr unsigned numThreads = 64;
    unsigned numBlocks            = iceil(leavesFocusIdx.size(), numThreads);
    if (numBlocks == 0) { return; }
    rangeCountKernel<<<numBlocks, numThreads>>>(leaves.data(), leaves.size(), counts.data(), leavesFocus.data(),
                                                leavesFocusIdx.data(), leavesFocusIdx.size(), countsFocus.data());
}

#define RANGE_COUNT_GPU(KeyType)                                                                                       \
    template void rangeCountGpu(std::span<const KeyType> leaves, std::span<const unsigned> counts,                     \
                                std::span<const KeyType> leavesFocus, std::span<const TreeNodeIndex> leavesFocusIdx,   \
                                std::span<unsigned> countsFocus)
RANGE_COUNT_GPU(uint32_t);
RANGE_COUNT_GPU(uint64_t);

} // namespace cstone
