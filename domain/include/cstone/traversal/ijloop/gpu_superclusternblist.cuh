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
 * @brief Neighbor search on GPU
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <tuple>
#include <type_traits>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/warp/warp_merge_sort.cuh>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include "cstone/compressneighbors.cuh"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/reducearray.cuh"
#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/traversal/ijloop/ijloop.hpp"
#include "cstone/tree/octree.hpp"

namespace cstone::ijloop
{

namespace gpu_supercluster_nb_list_neighborhood_detail
{

struct SuperclusterInfo
{
    unsigned index, neighborsCount, dataIndex;

    constexpr bool operator<(const SuperclusterInfo& other) const { return neighborsCount > other.neighborsCount; }
};

struct GlobalBuildData
{
    unsigned long long neighborDataSize;
    unsigned index;
};

constexpr __forceinline__ bool includeNbSymmetric(unsigned i, unsigned j, unsigned first, unsigned last)
{
    const bool s = i % 2 == j % 2;
    return (j < first) | (j >= last) | (i == j) | (i < j ? s : !s);
}

template<class Config>
constexpr __forceinline__ unsigned masksSize(unsigned numJClusters)
{
    return (numJClusters * Config::iClustersPerSupercluster * Config::numWarpsPerInteraction + 31) / 32;
}

template<class Config>
constexpr __forceinline__ unsigned superclusterIndex(unsigned i)
{
    return i / Config::superclusterSize;
}

template<class Config>
constexpr __forceinline__ unsigned jClusterIndex(unsigned j)
{
    return j / Config::jSize;
}

template<class Config>
constexpr __forceinline__ unsigned clusterOffset(unsigned firstBody)
{
    const unsigned offset =
        (firstBody + Config::superclusterSize - 1) / Config::superclusterSize * Config::superclusterSize - firstBody;
    assert(offset < Config::superclusterSize);
    return offset;
}

template<class T>
constexpr __forceinline__ void atomicAddScalarOrVec(T* ptr, T value)
{
    atomicAdd(ptr, value);
}

template<class T, std::size_t N>
constexpr __forceinline__ void atomicAddScalarOrVec(util::array<T, N>* ptr, util::array<T, N> const& value)
{
#pragma unroll
    for (std::size_t i = 0; i < N; ++i)
        atomicAddScalarOrVec(&((*ptr)[i]), value[i]);
}

__global__ void initSuperclusterInfo(const LocalIndex firstISupercluster,
                                     const LocalIndex lastISupercluster,
                                     SuperclusterInfo* superclusterInfo)
{
    const auto grid      = cooperative_groups::this_grid();
    const unsigned index = grid.thread_rank();

    const LocalIndex numISuperclusters = lastISupercluster - firstISupercluster;
    if (index < numISuperclusters) superclusterInfo[index] = {index + firstISupercluster, 0, 0};
}

template<class Config>
__global__ void
computeSuperclusterSplitMasks(const LocalIndex firstISupercluster,
                              const LocalIndex lastISupercluster,
                              const LocalIndex firstValidBody,
                              const GroupView __grid_constant__ groups,
                              typename Config::SuperclusterSplitMask* __restrict__ superclusterSplitMasks)
{
    const auto grid      = cooperative_groups::this_grid();
    const unsigned index = grid.thread_rank();
    if (index >= groups.numGroups) return;

    const LocalIndex groupEnd      = groups.groupEnd[index] + firstValidBody;
    const LocalIndex splitPosition = groupEnd % Config::superclusterSize;
    if (splitPosition == 0) return;

    const LocalIndex supercluster                       = groupEnd / Config::superclusterSize;
    auto* splitMaskPtr                                  = &superclusterSplitMasks[supercluster - firstISupercluster];
    typename Config::SuperclusterSplitMask oldSplitMask = *splitMaskPtr;
    typename Config::SuperclusterSplitMask newSplitMask;

    do
    {
        newSplitMask = oldSplitMask | (Config::SuperclusterSplitMask(1) << splitPosition);
        oldSplitMask = atomicCAS(splitMaskPtr, oldSplitMask, newSplitMask);
    } while (oldSplitMask != newSplitMask);
}

template<class Config, class Tc>
__global__ void computeJClusterBboxes(const LocalIndex firstValidBody,
                                      const LocalIndex totalBodies,
                                      const Tc* const __restrict__ x,
                                      const Tc* const __restrict__ y,
                                      const Tc* const __restrict__ z,
                                      Vec3<Tc>* const __restrict__ bboxCenters,
                                      Vec3<Tc>* const __restrict__ bboxSizes)
{
    static_assert(GpuConfig::warpSize % Config::jSize == 0);

    const auto block = cooperative_groups::this_thread_block();
    const auto warp  = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);

    const unsigned i = block.thread_index().x + block.group_dim().x * block.group_index().x;

    const Tc xi = x[std::max(std::min(i, totalBodies - 1), firstValidBody)];
    const Tc yi = y[std::max(std::min(i, totalBodies - 1), firstValidBody)];
    const Tc zi = z[std::max(std::min(i, totalBodies - 1), firstValidBody)];

    const unsigned numJClusters = jClusterIndex<Config>(totalBodies - 1) + 1;
    const unsigned jCluster     = jClusterIndex<Config>(i);

    if constexpr (Config::jSize >= 3)
    {
        util::array<Tc, 3> bboxMin{xi, yi, zi};
        util::array<Tc, 3> bboxMax{xi, yi, zi};

        const Tc vMin = reduceArray<Config::jSize, false>(bboxMin, [](auto a, auto b) { return std::min(a, b); });
        const Tc vMax = reduceArray<Config::jSize, false>(bboxMax, [](auto a, auto b) { return std::max(a, b); });

        const Tc center = (vMax + vMin) * Tc(0.5);
        const Tc size   = (vMax - vMin) * Tc(0.5);

        const unsigned idx = warp.thread_rank() % Config::jSize;
        if (idx < 3 & jCluster < numJClusters)
        {
            Tc* centerPtr = (Tc*)&bboxCenters[jCluster] + idx;
            Tc* sizePtr   = (Tc*)&bboxSizes[jCluster] + idx;
            *centerPtr    = center;
            *sizePtr      = size;
        }
    }
    else
    {
        Vec3<Tc> bboxMin{xi, yi, zi};
        Vec3<Tc> bboxMax{xi, yi, zi};

#pragma unroll
        for (unsigned offset = Config::jSize / 2; offset >= 1; offset /= 2)
        {
            bboxMin = {std::min(warp.shfl_down(bboxMin[0], offset), bboxMin[0]),
                       std::min(warp.shfl_down(bboxMin[1], offset), bboxMin[1]),
                       std::min(warp.shfl_down(bboxMin[2], offset), bboxMin[2])};
            bboxMax = {std::max(warp.shfl_down(bboxMax[0], offset), bboxMax[0]),
                       std::max(warp.shfl_down(bboxMax[1], offset), bboxMax[1]),
                       std::max(warp.shfl_down(bboxMax[2], offset), bboxMax[2])};
        }

        Vec3<Tc> center = (bboxMax + bboxMin) * Tc(0.5);
        Vec3<Tc> size   = (bboxMax - bboxMin) * Tc(0.5);

        if (i % Config::jSize == 0 && jCluster < numJClusters)
        {
            bboxCenters[jCluster] = center;
            bboxSizes[jCluster]   = size;
        }
    }
}

template<class Config, unsigned NumSuperclustersPerBlock, bool UsePbc, class Tc, class Th, class KeyType>
__device__ __forceinline__ void collectJClusterCandidates(const OctreeNsView<Tc, KeyType>& tree,
                                                          const Box<Tc>& box,
                                                          const LocalIndex firstValidBody,
                                                          const LocalIndex totalBodies,
                                                          const LocalIndex firstBody,
                                                          const LocalIndex lastBody,
                                                          const LocalIndex firstGroupParticle,
                                                          const LocalIndex lastGroupParticle,
                                                          const Tc* const __restrict__ x,
                                                          const Tc* const __restrict__ y,
                                                          const Tc* const __restrict__ z,
                                                          const Th* const __restrict__ h,
                                                          const Th maxH,
                                                          const Vec3<Tc>* const __restrict__ jClusterBboxCenters,
                                                          const Vec3<Tc>* const __restrict__ jClusterBboxSizes,
                                                          int* __restrict__ globalPool,
                                                          unsigned* candidates,
                                                          unsigned& numCandidates)
{
    const auto grid  = cooperative_groups::this_grid();
    const auto block = cooperative_groups::this_thread_block();
    const auto warp  = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);

    Vec3<Tc> bbMin = {std::numeric_limits<Tc>::max(), std::numeric_limits<Tc>::max(), std::numeric_limits<Tc>::max()};
    Vec3<Tc> bbMax = {std::numeric_limits<Tc>::lowest(), std::numeric_limits<Tc>::lowest(),
                      std::numeric_limits<Tc>::lowest()};

    for (unsigned i = firstGroupParticle + warp.thread_rank(); i < lastGroupParticle; i += warp.num_threads())
    {
        const Vec3<Tc> iPos = {x[i], y[i], z[i]};
        const Tc hBound     = Config::symmetric ? maxH : h[i];
#pragma unroll
        for (unsigned d = 0; d < 3; ++d)
        {
            bbMin[d] = std::min(bbMin[d], iPos[d] - 2 * hBound);
            bbMax[d] = std::max(bbMax[d], iPos[d] + 2 * hBound);
        }
    }
#pragma unroll
    for (unsigned d = 0; d < 3; ++d)
    {
        bbMin[d] = cooperative_groups::reduce(warp, bbMin[d], cooperative_groups::less<Tc>());
        bbMax[d] = cooperative_groups::reduce(warp, bbMax[d], cooperative_groups::greater<Tc>());
    }

    const Vec3<Tc> groupCenter = (bbMax + bbMin) * Tc(0.5);
    const Vec3<Tc> groupSize   = (bbMax - bbMin) * Tc(0.5);

    const unsigned firstISupercluster = superclusterIndex<Config>(firstBody);
    const unsigned lastISupercluster  = superclusterIndex<Config>(lastBody - 1) + 1;
    const unsigned iSupercluster      = superclusterIndex<Config>(firstGroupParticle);
    const unsigned numJClusters       = jClusterIndex<Config>(totalBodies - 1) + 1;

    const auto checkOverlap = [&](const unsigned jCluster, const unsigned numLanesValid)
    {
        assert(numLanesValid > 0);

        const unsigned prevJCluster = warp.shfl_up(jCluster, 1);
        bool isNeighbor             = warp.thread_rank() < numLanesValid & jCluster < numJClusters &
                          (warp.thread_rank() == 0 | prevJCluster != jCluster);

        if (isNeighbor)
        {
            isNeighbor = !Config::symmetric;
            if constexpr (Config::symmetric)
            {
                const unsigned jSupercluster = superclusterIndex<Config>(jCluster * Config::jSize);
                isNeighbor |= includeNbSymmetric(iSupercluster, jSupercluster, firstISupercluster, lastISupercluster);
            }

            if (isNeighbor)
            {
                const Vec3<Tc> jClusterCenter = jClusterBboxCenters[jCluster];
                const Vec3<Tc> jClusterSize   = jClusterBboxSizes[jCluster];
                isNeighbor &= cellOverlap<UsePbc>(jClusterCenter, jClusterSize, groupCenter, groupSize, box);
            }
        }

        const unsigned nbIndex = exclusiveScanBool(isNeighbor);
        // TODO: proper error handling
        assert(numCandidates + nbIndex < Config::ncMax);
        if (isNeighbor & (numCandidates + nbIndex < Config::ncMax)) candidates[numCandidates + nbIndex] = jCluster;
        numCandidates = warp.shfl(numCandidates + nbIndex + isNeighbor, GpuConfig::warpSize - 1);
    };

    volatile __shared__ int sharedPool[NumSuperclustersPerBlock * GpuConfig::warpSize];

    int jClusterQueue; // warp queue for source jCluster indices
    volatile int* tempQueue = sharedPool + GpuConfig::warpSize * warp.meta_group_rank();
    int* cellQueue =
        globalPool + TravConfig::memPerWarp * (grid.block_rank() * warp.meta_group_size() + warp.meta_group_rank());
    const TreeNodeIndex* __restrict__ childOffsets   = tree.childOffsets;
    const TreeNodeIndex* __restrict__ internalToLeaf = tree.internalToLeaf;
    const LocalIndex* __restrict__ layout            = tree.layout;
    const Vec3<Tc>* __restrict__ centers             = tree.centers;
    const Vec3<Tc>* __restrict__ sizes               = tree.sizes;

    // populate initial cell queue
    if (warp.thread_rank() == 0) cellQueue[0] = 1;
    warp.sync();

    // these variables are always identical on all warp lanes
    int numSources        = 1; // current stack size
    int newSources        = 0; // stack size for next level
    int oldSources        = 0; // cell indices done
    int sourceOffset      = 0; // current level stack pointer, once this reaches numSources, the level is done
    int jClusterFillLevel = 0; // fill level of the source jCluster warp queue

    while (numSources > 0) // While there are source cells to traverse
    {
        int sourceIdx   = sourceOffset + warp.thread_rank();
        int sourceQueue = 0;
        if (warp.thread_rank() < GpuConfig::warpSize / 8)
            sourceQueue = cellQueue[ringAddr(oldSources + sourceIdx)]; // Global source cell index in queue
        sourceQueue = spreadSeg8(sourceQueue);
        sourceIdx   = warp.shfl(sourceIdx, warp.thread_rank() / 8);

        const Vec3<Tc> curSrcCenter = centers[sourceQueue];      // Current source cell center
        const Vec3<Tc> curSrcSize   = sizes[sourceQueue];        // Current source cell center
        const int childBegin        = childOffsets[sourceQueue]; // First child cell
        const bool isNode           = childBegin;
        const bool isClose          = cellOverlap<UsePbc>(curSrcCenter, curSrcSize, groupCenter, groupSize, box);
        const bool isSource         = sourceIdx < numSources; // Source index is within bounds
        const bool isDirect         = isClose && !isNode && isSource;
        const int leafIdx           = isDirect ? internalToLeaf[sourceQueue] : 0; // the cstone leaf index

        // Split
        const bool isSplit     = isNode && isClose && isSource;                   // Source cell must be split
        const int numChildLane = exclusiveScanBool(isSplit);                      // Exclusive scan of numChild
        const int numChildWarp = reduceBool(isSplit);                             // Total numChild of current warp
        sourceOffset += imin(GpuConfig::warpSize / 8, numSources - sourceOffset); // advance current level stack pointer
        int childIdx = oldSources + numSources + newSources + numChildLane;       // Child index of current lane
        if (isSplit) cellQueue[ringAddr(childIdx)] = childBegin;                  // Queue child cells for next level
        newSources += numChildWarp; // Increment source cell count for next loop

        // check for cellQueue overflow
        const unsigned stackUsed = newSources + numSources - sourceOffset; // current cellQueue size
        if (stackUsed > TravConfig::memPerWarp) return;                    // Exit if cellQueue overflows

        // Direct
        const int firstJCluster = jClusterIndex<Config>(layout[leafIdx] + firstValidBody);
        const int numJClusters = (jClusterIndex<Config>(layout[leafIdx + 1] + firstValidBody - 1) + 1 - firstJCluster) &
                                 -int(isDirect); // Number of jClusters in cell
        bool directTodo            = numJClusters;
        const int numJClustersScan = inclusiveScanInt(numJClusters);  // Inclusive scan of numJClusters
        int numJClustersLane       = numJClustersScan - numJClusters; // Exclusive scan of numJClusters
        int numJClustersWarp =
            shflSync(numJClustersScan, GpuConfig::warpSize - 1); // Total numJClusters of current warp
        int prevJClusterIdx = 0;
        while (numJClustersWarp > 0) // While there are jClusters to process from current source cell set
        {
            tempQueue[warp.thread_rank()] =
                1; // Default scan input is 1, such that consecutive lanes load consecutive bodies
            if (directTodo && (numJClustersLane < GpuConfig::warpSize))
            {
                directTodo                  = false;              // Set cell as processed
                tempQueue[numJClustersLane] = -1 - firstJCluster; // Put first source cell body index into the queue
            }
            const int jClusterIdx = inclusiveSegscanInt(tempQueue[warp.thread_rank()], prevJClusterIdx);
            // broadcast last processed jClusterIdx from the last lane to restart the scan in the next iteration
            prevJClusterIdx = shflSync(jClusterIdx, GpuConfig::warpSize - 1);

            if (numJClustersWarp >= GpuConfig::warpSize) // Process jClusters from current set of source cells
            {
                checkOverlap(jClusterIdx, GpuConfig::warpSize);
                numJClustersWarp -= GpuConfig::warpSize;
                numJClustersLane -= GpuConfig::warpSize;
            }
            else // Fewer than warpSize bodies remaining from current source cell set
            {
                // push the remaining bodies into jClusterQueue
                int topUp     = shflUpSync(jClusterIdx, jClusterFillLevel);
                jClusterQueue = (warp.thread_rank() < jClusterFillLevel) ? jClusterQueue : topUp;

                jClusterFillLevel += numJClustersWarp;
                if (jClusterFillLevel >= GpuConfig::warpSize) // If this causes jClusterQueue to spill
                {
                    checkOverlap(jClusterQueue, GpuConfig::warpSize);
                    jClusterFillLevel -= GpuConfig::warpSize;
                    // jClusterQueue is now empty; put body indices that spilled into the queue
                    jClusterQueue = shflDownSync(jClusterIdx, numJClustersWarp - jClusterFillLevel);
                }
                numJClustersWarp = 0; // No more bodies to process from current source cells
            }
        }

        //  If the current level is done
        if (sourceOffset >= numSources)
        {
            oldSources += numSources;      // Update finished source size
            numSources   = newSources;     // Update current source size
            sourceOffset = newSources = 0; // Initialize next source size and offset
        }
    }

    if (jClusterFillLevel > 0) // If there are leftover direct bodies
        checkOverlap(jClusterQueue, jClusterFillLevel);
}

template<class Config, unsigned NumSuperclustersPerBlock>
__device__ __forceinline__ void sortCandidates(std::uint32_t* candidates, unsigned numCandidates)
{
    const auto block = cooperative_groups::this_thread_block();
    const auto warp  = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);

    constexpr unsigned itemsPerWarp = Config::ncMax / GpuConfig::warpSize;
    std::uint32_t items[itemsPerWarp];
#pragma unroll
    for (unsigned i = 0; i < itemsPerWarp; ++i)
    {
        const unsigned c = warp.thread_rank() * itemsPerWarp + i;
        items[i]         = c < numCandidates ? candidates[c] : std::numeric_limits<std::uint32_t>::max();
    }

    using WarpSort = cub::WarpMergeSort<std::uint32_t, itemsPerWarp, GpuConfig::warpSize>;
    __shared__ typename WarpSort::TempStorage sortTmp[NumSuperclustersPerBlock];
    WarpSort(sortTmp[warp.meta_group_rank()]).Sort(items, std::less<unsigned>());

#pragma unroll
    for (unsigned i = 0; i < itemsPerWarp; ++i)
    {
        const unsigned c = warp.thread_rank() * itemsPerWarp + i;
        if (c < numCandidates) candidates[c] = items[i];
    }

    warp.sync();
}

__device__ __forceinline__ void pruneCandidates(std::uint32_t* __restrict__ jClusters, unsigned& numCandidates)
{
    const auto block = cooperative_groups::this_thread_block();
    const auto warp  = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);

    unsigned prunedCandidates = 0;
    std::uint32_t previous    = ~0u;

    for (unsigned n = 0; n < numCandidates; n += warp.num_threads())
    {
        const std::uint32_t candidate   = jClusters[std::min(n + warp.thread_rank(), numCandidates - 1)];
        std::uint32_t previousCandidate = warp.shfl_up(candidate, 1);
        if (warp.thread_rank() == 0) previousCandidate = previous;

        const bool keep      = candidate != previousCandidate;
        const unsigned index = exclusiveScanBool(keep);
        if (keep) jClusters[prunedCandidates + index] = candidate;
        prunedCandidates = warp.shfl(prunedCandidates + index + keep, warp.num_threads() - 1);
        previous         = warp.shfl(candidate, warp.num_threads() - 1);
    }

    numCandidates = prunedCandidates;
}

template<class Config, unsigned NumSuperclustersPerBlock, bool UsePbc, class Tc, class Th>
__device__ __forceinline__ void pruneCandidatesAndComputeMasks(const Box<Tc>& box,
                                                               const LocalIndex firstValidBody,
                                                               const LocalIndex totalBodies,
                                                               const Tc* const __restrict__ x,
                                                               const Tc* const __restrict__ y,
                                                               const Tc* const __restrict__ z,
                                                               const Th* const __restrict__ h,
                                                               const unsigned iSupercluster,
                                                               std::uint32_t* __restrict__ jClusters,
                                                               std::uint32_t* __restrict__ masks,
                                                               const unsigned numCandidates,
                                                               unsigned& numJClusters)
{
    const auto block = cooperative_groups::this_thread_block();
    const auto warp  = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);

    __shared__ Tc xisBuffer[NumSuperclustersPerBlock][Config::superclusterSize];
    __shared__ Tc yisBuffer[NumSuperclustersPerBlock][Config::superclusterSize];
    __shared__ Tc zisBuffer[NumSuperclustersPerBlock][Config::superclusterSize];
    __shared__ Th hisBuffer[NumSuperclustersPerBlock][Config::superclusterSize];
    Tc* xis = xisBuffer[warp.meta_group_rank()];
    Tc* yis = yisBuffer[warp.meta_group_rank()];
    Tc* zis = zisBuffer[warp.meta_group_rank()];
    Th* his = hisBuffer[warp.meta_group_rank()];

    for (unsigned n = warp.thread_rank(); n < Config::superclusterSize; n += warp.num_threads())
    {
        const unsigned i =
            std::max(std::min(Config::superclusterSize * iSupercluster + n, totalBodies - 1), firstValidBody);
        xis[n] = x[i];
        yis[n] = y[i];
        zis[n] = z[i];
        his[n] = h[i];
    }

    const unsigned maxMasksSize = masksSize<Config>(numCandidates);
    for (unsigned n = warp.thread_rank(); n < maxMasksSize; n += warp.num_threads())
        masks[n] = 0;

    warp.sync();

    constexpr unsigned iClustersPerWarp = Config::iThreads / Config::iSize;
    const unsigned iClusterOffset       = iClustersPerWarp == 1 ? 0 : block.thread_index().x / Config::iSize;

    std::uint32_t previousJCluster = std::numeric_limits<std::uint32_t>::max();
    numJClusters                   = 0;
    for (unsigned candidate = 0; candidate < numCandidates; ++candidate)
    {
        const std::uint32_t jCluster = jClusters[candidate];
        if (jCluster == previousJCluster) continue;
        previousJCluster = jCluster;

        std::uint32_t mask = 0;
        for (unsigned w = 0; w < Config::numWarpsPerInteraction; ++w)
        {
            const unsigned j = jCluster * Config::jSize + (Config::jSize / Config::numWarpsPerInteraction) * w +
                               block.thread_index().y;
            const unsigned jSupercluster = superclusterIndex<Config>(j);
            if (j >= firstValidBody & j < totalBodies)
            {
                const Tc xj = x[j];
                const Tc yj = y[j];
                const Tc zj = z[j];
                const Th hj = h[j];

                for (unsigned c = 0; c < Config::iClustersPerSupercluster; c += iClustersPerWarp)
                {
                    const unsigned ci = c + iClusterOffset;
                    const unsigned i  = ci * Config::iSize + block.thread_index().x % Config::iSize;
                    if (!Config::symmetric | (iSupercluster != jSupercluster) | (i <= j))
                    {
                        const unsigned si = ci * Config::iSize + block.thread_index().x % Config::iSize;
                        const Tc xi       = xis[si];
                        const Tc yi       = yis[si];
                        const Tc zi       = zis[si];
                        const Th hi       = his[si];
                        Tc xij            = xi - xj;
                        Tc yij            = yi - yj;
                        Tc zij            = zi - zj;
                        if constexpr (UsePbc)
                        {
                            xij -= (box.boundaryX() == BoundaryType::periodic) * box.lx() * std::rint(xij * box.ilx());
                            yij -= (box.boundaryY() == BoundaryType::periodic) * box.ly() * std::rint(yij * box.ily());
                            zij -= (box.boundaryZ() == BoundaryType::periodic) * box.lz() * std::rint(zij * box.ilz());
                        }
                        const Th distSq             = xij * xij + yij * yij + zij * zij;
                        const Th hMax               = Config::symmetric ? std::max(hi, hj) : hi;
                        const bool overlaps         = distSq < Th(4) * hMax * hMax;
                        const unsigned maskBitIndex = w * Config::iClustersPerSupercluster + ci;
                        assert(maskBitIndex < 32);
                        mask |= std::uint32_t(overlaps) << maskBitIndex;
                    }
                }
            }
        }
        mask = cooperative_groups::reduce(warp, mask, cooperative_groups::bit_or<std::uint32_t>());
        if (mask)
        {
            if (warp.thread_rank() == 0)
            {
                const unsigned maskStartIndex =
                    numJClusters * (Config::iClustersPerSupercluster * Config::numWarpsPerInteraction);
                masks[maskStartIndex / 32] |= mask << (maskStartIndex % 32);
                jClusters[numJClusters] = jCluster;
            }
            ++numJClusters;
        }
    }
}

template<class Config, unsigned NumSuperclustersPerBlock>
__device__ __forceinline__ void storeNeighborData(const std::uint32_t* const __restrict__ jClusters,
                                                  const std::uint32_t* const __restrict__ masks,
                                                  std::uint32_t* const __restrict__ neighborData,
                                                  const unsigned neighborDataSize,
                                                  SuperclusterInfo& info,
                                                  GlobalBuildData* __restrict__ globalBuildData)
{
    const auto block = cooperative_groups::this_thread_block();
    const auto warp  = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);

    const unsigned mSize = masksSize<Config>(info.neighborsCount);
    unsigned nbSize      = info.neighborsCount;

    __shared__ std::uint32_t compressedJClusters[NumSuperclustersPerBlock][Config::compress ? Config::ncMax : 1];

    if constexpr (Config::compress)
    {
        warpCompressNeighbors(jClusters, (char*)compressedJClusters[warp.meta_group_rank()], info.neighborsCount);
        nbSize = compressedNeighborsSize((const char*)compressedJClusters[warp.meta_group_rank()]);
    }

    const unsigned long long totalSize = nbSize + mSize;
    if (warp.thread_rank() == 0) info.dataIndex = atomicAdd(&globalBuildData->neighborDataSize, totalSize);
    info.dataIndex = warp.shfl(info.dataIndex, 0);

    for (unsigned n = warp.thread_rank(); n < mSize; n += warp.num_threads())
    {
        const auto index = info.dataIndex + n;
        if (index < neighborDataSize) neighborData[index] = masks[n];
    }

    for (unsigned n = warp.thread_rank(); n < nbSize; n += warp.num_threads())
    {
        const auto index = info.dataIndex + mSize + n;
        if (index < neighborDataSize)
            neighborData[index] = (Config::compress ? compressedJClusters[warp.meta_group_rank()] : jClusters)[n];
    }
}

template<class Config, unsigned NumSuperclustersPerBlock, bool UsePbc, class Tc, class Th, class KeyType>
__global__ __launch_bounds__(GpuConfig::warpSize* NumSuperclustersPerBlock) void buildNbList(
    const OctreeNsView<Tc, KeyType> __grid_constant__ tree,
    const Box<Tc> __grid_constant__ box,
    const LocalIndex firstValidBody,
    const LocalIndex totalBodies,
    const LocalIndex firstBody,
    const LocalIndex lastBody,
    const Tc* const __restrict__ x,
    const Tc* const __restrict__ y,
    const Tc* const __restrict__ z,
    const Th* const __restrict__ h,
    const Th maxH,
    const Vec3<Tc>* const __restrict__ jClusterBboxCenters,
    const Vec3<Tc>* const __restrict__ jClusterBboxSizes,
    const typename Config::SuperclusterSplitMask* const __restrict__ superclusterSplitMasks,
    std::uint32_t* const __restrict__ neighborData,
    const std::size_t neighborDataSize,
    SuperclusterInfo* const __restrict__ superclusterInfo,
    const unsigned numSuperClusters,
    int* __restrict__ globalPool,
    GlobalBuildData* __restrict__ globalBuildData)
{
    const auto block = cooperative_groups::this_thread_block();
    const auto warp  = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);

    assert(block.dim_threads().x == Config::iThreads);
    assert(block.dim_threads().y == GpuConfig::warpSize / Config::iThreads);
    assert(block.dim_threads().z == NumSuperclustersPerBlock);

    while (true)
    {
        unsigned index;
        if (warp.thread_rank() == 0) index = atomicAdd(&globalBuildData->index, 1);
        index = warp.shfl(index, 0);
        if (index >= numSuperClusters) return;

        SuperclusterInfo info = {.index = superclusterInfo[index].index, .neighborsCount = 0, .dataIndex = 0};

        __shared__ std::uint32_t jClustersBuffer[NumSuperclustersPerBlock][Config::ncMax];
        std::uint32_t* jClusters = jClustersBuffer[warp.meta_group_rank()];

        const unsigned firstISupercluster = superclusterIndex<Config>(firstBody);
        auto splitMask                    = superclusterSplitMasks[info.index - firstISupercluster];
        assert(!(splitMask & 1));
        unsigned numCandidates = 0;

        unsigned firstGroupParticle       = std::max(info.index * Config::superclusterSize, firstValidBody);
        unsigned lastGroupParticle        = firstGroupParticle;
        const unsigned finalGroupParticle = std::min((info.index + 1) * Config::superclusterSize, totalBodies);
        while (lastGroupParticle < finalGroupParticle)
        {
            firstGroupParticle = lastGroupParticle;
            do
            {
                ++lastGroupParticle;
            } while (!((splitMask >>= 1) & 1) & (lastGroupParticle < finalGroupParticle));

            collectJClusterCandidates<Config, NumSuperclustersPerBlock, UsePbc>(
                tree, box, firstValidBody, totalBodies, firstBody, lastBody, firstGroupParticle, lastGroupParticle, x,
                y, z, h, maxH, jClusterBboxCenters, jClusterBboxSizes, globalPool, jClusters, numCandidates);

            sortCandidates<Config, NumSuperclustersPerBlock>(jClusters, numCandidates);
            if (lastGroupParticle < finalGroupParticle) pruneCandidates(jClusters, numCandidates);
        }

        __shared__ std::uint32_t masksBuffer[NumSuperclustersPerBlock][masksSize<Config>(Config::ncMax)];
        std::uint32_t* masks = masksBuffer[warp.meta_group_rank()];

        pruneCandidatesAndComputeMasks<Config, NumSuperclustersPerBlock, UsePbc>(box, firstValidBody, totalBodies, x, y,
                                                                                 z, h, info.index, jClusters, masks,
                                                                                 numCandidates, info.neighborsCount);

        storeNeighborData<Config, NumSuperclustersPerBlock>(jClusters, masks, neighborData, neighborDataSize, info,
                                                            globalBuildData);

        if (warp.thread_rank() == 0) superclusterInfo[index] = info;
    }
}

template<class T0, class... T>
__device__ inline constexpr T0 dynamicTupleGet(std::tuple<T0, T...> const& tuple, int index)
{
    T0 res;
    int i = 0;
    util::for_each_tuple(
        [&](auto const& src)
        {
            if (i++ == index) res = src;
        },
        tuple);
    return res;
}

template<class Config, class T0, class... T>
__device__ __forceinline__ void
storeTupleISum(std::tuple<T0, T...> tuple, std::tuple<T0*, T*...> const& ptrs, const unsigned index, const bool store)
{
    const auto block = cooperative_groups::this_thread_block();
    assert(block.dim_threads().x == Config::iThreads);
    const auto warp = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);

    if constexpr (std::conjunction_v<std::is_same<T0, T>...> && sizeof...(T) < GpuConfig::warpSize / Config::iThreads)
    {
        const T0 res = reduceTuple<GpuConfig::warpSize / Config::iThreads, true>(tuple, std::plus<T0>());
        if ((block.thread_index().y % (GpuConfig::warpSize / Config::iThreads) <= sizeof...(T)) & store)
        {
            T0* ptr = dynamicTupleGet(ptrs, block.thread_index().y % (GpuConfig::warpSize / Config::iThreads));
            if constexpr (Config::symmetric | (Config::numWarpsPerInteraction > 1))
                atomicAddScalarOrVec(&ptr[index], res);
            else
                ptr[index] = res;
        }
    }
    else
    {
#pragma unroll
        for (unsigned offset = GpuConfig::warpSize / 2; offset >= Config::iThreads; offset /= 2)
            util::for_each_tuple([&](auto& t) { t += warp.shfl_down(t, offset); }, tuple);

        if ((block.thread_index().y % (GpuConfig::warpSize / Config::iThreads) == 0) & store)
            util::for_each_tuple(
                [index](auto* ptr, auto const& t)
                {
                    if constexpr (Config::symmetric | (Config::numWarpsPerInteraction > 1))
                    {
                        atomicAddScalarOrVec(&ptr[index], t);
                    }
                    else { ptr[index] = t; }
                },
                ptrs, tuple);
    }
}

template<class Config, class T0, class... T>
constexpr __device__ void
storeTupleJSum(std::tuple<T0, T...> tuple, std::tuple<T0*, T*...> const& ptrs, const unsigned index, const bool store)
{
    const auto block = cooperative_groups::this_thread_block();
    assert(block.dim_threads().x == Config::iThreads);
    const auto warp = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);

    if constexpr (std::conjunction_v<std::is_same<T0, T>...> && sizeof...(T) < Config::iThreads)
    {
        const T0 res = reduceTuple<Config::iThreads, false>(tuple, std::plus<T0>());
        if ((block.thread_index().x <= sizeof...(T)) & store)
        {
            T0* ptr = dynamicTupleGet(ptrs, block.thread_index().x);
            atomicAddScalarOrVec(&ptr[index], res);
        }
    }
    else
    {
#pragma unroll
        for (unsigned offset = Config::iThreads / 2; offset >= 1; offset /= 2)
            util::for_each_tuple([&](auto& t) { t += warp.shfl_down(t, offset); }, tuple);

        if ((block.thread_index().x == 0) & store)
            util::for_each_tuple([index](auto* ptr, auto const& t) { atomicAddScalarOrVec(&ptr[index], t); }, ptrs,
                                 tuple);
    }
}

template<class Config,
         unsigned NumSuperclustersPerBlock,
         bool UsePbc,
         Symmetry Sym,
         class Tc,
         class Th,
         class In,
         class Out,
         class Interaction>
__global__ __launch_bounds__(Config::iThreads* Config::jSize* NumSuperclustersPerBlock) void runIjLoop(
    const Box<Tc> __grid_constant__ box,
    const LocalIndex firstValidBody,
    const LocalIndex totalBodies,
    const LocalIndex firstBody,
    const LocalIndex lastBody,
    const Tc* const __restrict__ x,
    const Tc* const __restrict__ y,
    const Tc* const __restrict__ z,
    const Th* const __restrict__ h,
    const In __grid_constant__ input,
    const Out __grid_constant__ output,
    const Interaction interaction,
    const std::uint32_t* const __restrict__ neighborData,
    const SuperclusterInfo* const __restrict__ superclusterInfo)
{
    static_assert(Config::ncMax % GpuConfig::warpSize == 0);
    static_assert(NumSuperclustersPerBlock > 0);
    static_assert(Config::iThreads * Config::jSize >= GpuConfig::warpSize);
    static_assert(Config::iThreads * Config::jSize % GpuConfig::warpSize == 0);

    const auto block = cooperative_groups::this_thread_block();
    assert(block.dim_threads().x == Config::iThreads);
    assert(block.dim_threads().y == Config::jSize);
    assert(block.dim_threads().z == NumSuperclustersPerBlock);

    const unsigned firstISupercluster = superclusterIndex<Config>(firstBody);
    const unsigned lastISupercluster  = superclusterIndex<Config>(lastBody - 1) + 1;
    const unsigned numISuperclusters  = lastISupercluster - firstISupercluster;
    const unsigned iSuperclusterIndex = block.group_index().x * NumSuperclustersPerBlock + block.thread_index().z;
    if (iSuperclusterIndex >= numISuperclusters) return;

    auto [iSupercluster, iSuperclusterNeighborsCount, iSuperclusterDataIndex] = superclusterInfo[iSuperclusterIndex];

    using particleData_t = decltype(loadParticleData(x, y, z, h, input, 0));

    // TODO: bank-conflict friendly SoA layout?
    __shared__ particleData_t
        iSuperclusterDataBuffer[NumSuperclustersPerBlock][Config::iClustersPerSupercluster * Config::iSize];
    particleData_t* iSuperclusterData = iSuperclusterDataBuffer[block.thread_index().z];
    {
        const unsigned base = iSupercluster * Config::superclusterSize;
        for (unsigned offset = block.thread_index().y * Config::iThreads + block.thread_index().x;
             offset < Config::iClustersPerSupercluster * Config::iSize; offset += Config::iThreads * Config::jSize)
        {
            const unsigned i = base + offset;
            auto iData       = (i >= firstValidBody & i < totalBodies) ? loadParticleData(x, y, z, h, input, i)
                                                                       : dummyParticleData(x, y, z, h, input, i);
            std::get<0>(iData) -= firstValidBody;
            iSuperclusterData[offset] = iData;
        }
    }

    __shared__ unsigned nbDataBuffer[NumSuperclustersPerBlock][Config::ncMax + masksSize<Config>(Config::ncMax)];
    unsigned* const nbData = nbDataBuffer[block.thread_index().z];

    const unsigned maskSize   = masksSize<Config>(iSuperclusterNeighborsCount);
    const unsigned nbDataSize = iSuperclusterNeighborsCount + maskSize;

    constexpr unsigned iClustersPerWarp = Config::iThreads / Config::iSize;
    const unsigned warpIndex            = block.thread_index().y / (Config::jSize / Config::numWarpsPerInteraction);

    if constexpr (Config::compress)
    {
        for (unsigned n = block.thread_index().y * Config::iThreads + block.thread_index().x; n < maskSize;
             n += Config::iThreads * Config::jSize)
            nbData[n] = neighborData[iSuperclusterDataIndex + n];
        // TODO: use all warps?
        if (warpIndex == 0)
        {
            unsigned n;
            warpDecompressNeighbors((const char*)&neighborData[iSuperclusterDataIndex + maskSize], &nbData[maskSize],
                                    n);
            assert(n == iSuperclusterNeighborsCount);
        }
    }
    else
    {
        for (unsigned n = block.thread_index().y * Config::iThreads + block.thread_index().x; n < nbDataSize;
             n += Config::iThreads * Config::jSize)
            nbData[n] = neighborData[iSuperclusterDataIndex + n];
    }

    block.sync();

    using result_t = decltype(interaction(particleData_t(), particleData_t(), Vec3<Tc>(), Tc(0)));
    std::array<result_t, Config::iClustersPerSupercluster / iClustersPerWarp> iResults = {};
    const unsigned iClusterOffset = iClustersPerWarp == 1 ? 0 : block.thread_index().x / Config::iSize;

    for (unsigned nb = 0; nb < iSuperclusterNeighborsCount; ++nb)
    {
        const unsigned maskStartIndex = nb * (Config::iClustersPerSupercluster * Config::numWarpsPerInteraction) +
                                        (warpIndex * Config::iClustersPerSupercluster);
        unsigned warpMask =
            (nbData[maskStartIndex / 32] >> (maskStartIndex % 32)) & ((1 << Config::iClustersPerSupercluster) - 1);

        if (warpMask)
        {
            const unsigned jCluster      = nb < iSuperclusterNeighborsCount ? nbData[nb + maskSize] : ~0u;
            const unsigned j             = jCluster * Config::jSize + block.thread_index().y;
            const unsigned jSupercluster = superclusterIndex<Config>(j);
            auto jData                   = (nb < iSuperclusterNeighborsCount & j >= firstValidBody & j < totalBodies)
                                               ? loadParticleData(x, y, z, h, input, j)
                                               : dummyParticleData(x, y, z, h, input, j);
            std::get<0>(jData) -= firstValidBody;
            result_t jResult = {};

            warpMask >>= iClusterOffset;
            unsigned i = iSupercluster * Config::superclusterSize + block.thread_index().x;
            for (unsigned c = 0; c < Config::iClustersPerSupercluster; c += iClustersPerWarp)
            {
                if ((warpMask & 1) && (!Config::symmetric | (iSupercluster != jSupercluster) | (i <= j)))
                {
                    const auto& iData = iSuperclusterData[c * Config::iSize + block.thread_index().x];
                    assert(std::get<0>(iData) == i - firstValidBody);
                    const auto [ijPosDiff, distSq] = posDiffAndDistSq(UsePbc, box, iData, jData);
                    auto ijInteraction             = interaction(iData, jData, ijPosDiff, distSq);
                    if (distSq < radiusSq(iData)) updateResult(iResults[c / iClustersPerWarp], ijInteraction);
                    if constexpr (Config::symmetric)
                    {
                        if ((distSq < radiusSq(jData)) & ((i != j) | (i < firstBody) | (i >= lastBody)))
                        {
                            if constexpr (std::is_same_v<Sym, symmetry::Asymmetric>)
                                ijInteraction = interaction(jData, iData, -ijPosDiff, distSq);
                            updateResult(jResult, ijInteraction);
                        }
                    }
                }
                warpMask >>= iClustersPerWarp;
                i += Config::iThreads;
            }

            if constexpr (Config::symmetric)
            {
                if constexpr (std::is_same_v<Sym, symmetry::Odd>)
                    util::for_each_tuple([](auto& v) { v = -v; }, jResult);

                storeTupleJSum<Config>(jResult, output, j, j >= firstBody & j < lastBody);
            }
        }
    }

    for (unsigned c = 0; c < Config::iClustersPerSupercluster; c += iClustersPerWarp)
    {
        const unsigned ci = c + iClusterOffset;
        const auto i =
            iSupercluster * Config::superclusterSize + ci * Config::iSize + block.thread_index().x % Config::iSize;
        storeTupleISum<Config>(iResults[c / iClustersPerWarp], output, i, i >= firstBody & i < lastBody);
    }
}

template<class Config, class Tc, class Th>
struct GpuSuperclusterNbListNeighborhoodImpl
{
    Box<Tc> box;
    LocalIndex firstValidBody, totalBodies, firstBody, lastBody;
    const Tc *x, *y, *z;
    const Th* h;
    thrust::device_vector<std::uint32_t> neighborData;
    thrust::device_vector<SuperclusterInfo> superclusterInfo;

    template<class... In, class... Out, class Interaction, Symmetry Sym>
    void ijLoop(std::tuple<In*...> input, std::tuple<Out*...> output, Interaction&& interaction, Sym) const
    {
        const LocalIndex numBodies = lastBody - firstBody;
        if (numBodies == 0) return;

        util::for_each_tuple([&](auto& ptr) { ptr -= firstValidBody; }, input);
        util::for_each_tuple([&](auto& ptr) { ptr -= firstValidBody; }, output);

        if (Config::symmetric | (Config::numWarpsPerInteraction > 1))
        {
            util::for_each_tuple(
                [&](auto* ptr)
                { checkGpuErrors(cudaMemsetAsync(ptr + firstBody, 0, sizeof(decltype(*ptr)) * numBodies)); }, output);
        }

        assert(firstBody < lastBody);
        const LocalIndex firstISupercluster = superclusterIndex<Config>(firstBody);
        const LocalIndex lastISupercluster  = superclusterIndex<Config>(lastBody - 1) + 1;
        const LocalIndex numISuperclusters  = lastISupercluster - firstISupercluster;

        constexpr unsigned numSuperclustersPerBlock = 64 / (Config::iThreads * Config::jSize);
        const dim3 blockSize                        = {Config::iThreads, Config::jSize, numSuperclustersPerBlock};
        const unsigned numBlocks                    = iceil(numISuperclusters, numSuperclustersPerBlock);
        const auto runKernel                        = [&](auto usePbc)
        {
            runIjLoop<Config, numSuperclustersPerBlock, decltype(usePbc)::value, Sym><<<numBlocks, blockSize>>>(
                box, firstValidBody, totalBodies, firstBody, lastBody, x, y, z, h, makeConstRestrict(input), output,
                std::forward<Interaction>(interaction), rawPtr(neighborData), rawPtr(superclusterInfo));
            checkGpuErrors(cudaGetLastError());
        };
        if (box.boundaryX() == BoundaryType::periodic | box.boundaryY() == BoundaryType::periodic |
            box.boundaryZ() == BoundaryType::periodic)
            runKernel(std::true_type());
        else
            runKernel(std::false_type());
    }

    Statistics stats() const
    {
        return {.numBodies = lastBody - firstBody,
                .numBytes  = neighborData.size() * sizeof(typename decltype(neighborData)::value_type) +
                            superclusterInfo.size() * sizeof(typename decltype(superclusterInfo)::value_type)};
    }
};

template<unsigned NcMax            = 256,
         unsigned ISize            = 8,
         unsigned JSize            = 8,
         unsigned SuperclusterSize = ISize * std::max(JSize, GpuConfig::warpSize / ISize),
         bool Compress             = false,
         bool Symmetric            = true>
struct GpuSuperclusterNbListNeighborhoodConfig
{
    static_assert((ISize & (ISize - 1)) == 0, "ISize must be power of two");
    static_assert((JSize & (JSize - 1)) == 0, "JSize must be power of two");
    static_assert(SuperclusterSize % ISize == 0, "SuperclusterSize must be divisible by ISize");
    static_assert(SuperclusterSize % JSize == 0, "SuperclusterSize must be divisible by JSize");

    static constexpr unsigned ncMax            = NcMax;
    static constexpr unsigned iSize            = ISize;
    static constexpr unsigned jSize            = JSize;
    static constexpr unsigned superclusterSize = SuperclusterSize;
    static constexpr bool compress             = Compress;
    static constexpr bool symmetric            = Symmetric;

    static constexpr unsigned iClustersPerSupercluster = superclusterSize / iSize;
    static constexpr unsigned iThreads                 = std::max(iSize, GpuConfig::warpSize / jSize);
    static constexpr unsigned numWarpsPerInteraction = (iSize * jSize + GpuConfig::warpSize - 1) / GpuConfig::warpSize;

    template<unsigned NewNcMax>
    using withNcMax =
        GpuSuperclusterNbListNeighborhoodConfig<NewNcMax, ISize, JSize, SuperclusterSize, Compress, Symmetric>;

    template<unsigned NewISize, unsigned NewJSize>
    using withClusterSize =
        GpuSuperclusterNbListNeighborhoodConfig<NcMax, NewISize, NewJSize, SuperclusterSize, Compress, Symmetric>;
    template<unsigned NewSuperclusterSize>
    using withSuperclusterSize =
        GpuSuperclusterNbListNeighborhoodConfig<NcMax, ISize, JSize, NewSuperclusterSize, Compress, Symmetric>;
    using withCompression =
        GpuSuperclusterNbListNeighborhoodConfig<NcMax, ISize, JSize, SuperclusterSize, true, Symmetric>;
    using withoutCompression =
        GpuSuperclusterNbListNeighborhoodConfig<NcMax, ISize, JSize, SuperclusterSize, false, Symmetric>;
    using withSymmetry = GpuSuperclusterNbListNeighborhoodConfig<NcMax, ISize, JSize, SuperclusterSize, Compress, true>;
    using withoutSymmetry =
        GpuSuperclusterNbListNeighborhoodConfig<NcMax, ISize, JSize, SuperclusterSize, Compress, false>;

    using SuperclusterSplitMask = std::conditional_t<(superclusterSize > 32), unsigned long long, unsigned>;
    static_assert(superclusterSize <= 64, "superclusters with more than 64 particles are not supported");
};

} // namespace gpu_supercluster_nb_list_neighborhood_detail

template<class Config = gpu_supercluster_nb_list_neighborhood_detail::GpuSuperclusterNbListNeighborhoodConfig<>>
struct GpuSuperclusterNbListNeighborhood
{
    template<unsigned NcMax>
    using withNcMax = GpuSuperclusterNbListNeighborhood<typename Config::template withNcMax<NcMax>>;
    template<unsigned ISize, unsigned JSize>
    using withClusterSize = GpuSuperclusterNbListNeighborhood<typename Config::template withClusterSize<ISize, JSize>>;
    template<unsigned SuperclusterSize>
    using withSuperclusterSize =
        GpuSuperclusterNbListNeighborhood<typename Config::template withSuperclusterSize<SuperclusterSize>>;
    using withCompression    = GpuSuperclusterNbListNeighborhood<typename Config::withCompression>;
    using withoutCompression = GpuSuperclusterNbListNeighborhood<typename Config::withoutCompression>;
    using withSymmetry       = GpuSuperclusterNbListNeighborhood<typename Config::withSymmetry>;
    using withoutSymmetry    = GpuSuperclusterNbListNeighborhood<typename Config::withoutSymmetry>;

    template<class Tc, class KeyType, class Th>
    gpu_supercluster_nb_list_neighborhood_detail::GpuSuperclusterNbListNeighborhoodImpl<Config, Tc, Th>
    build(const OctreeNsView<Tc, KeyType>& tree,
          const Box<Tc>& box,
          LocalIndex totalBodies,
          GroupView groups,
          const Tc* x,
          const Tc* y,
          const Tc* z,
          const Th* h) const
    {
        using namespace gpu_supercluster_nb_list_neighborhood_detail;

        const LocalIndex firstValidBody = clusterOffset<Config>(groups.firstBody);
        groups.firstBody += firstValidBody;
        assert(groups.firstBody % Config::superclusterSize == 0);
        groups.lastBody += firstValidBody;
        totalBodies += firstValidBody;
        x -= firstValidBody;
        y -= firstValidBody;
        z -= firstValidBody;
        h -= firstValidBody;

        const LocalIndex firstISupercluster = superclusterIndex<Config>(groups.firstBody);
        const LocalIndex lastISupercluster  = superclusterIndex<Config>(groups.lastBody - 1) + 1;
        const LocalIndex numISuperclusters  = lastISupercluster - firstISupercluster;
        const LocalIndex numJClusters       = jClusterIndex<Config>(totalBodies - 1) + 1;

        GpuSuperclusterNbListNeighborhoodImpl<Config, Tc, Th> nbList{
            box,
            firstValidBody,
            totalBodies,
            groups.firstBody,
            groups.lastBody,
            x,
            y,
            z,
            h,
            thrust::device_vector<std::uint32_t>(),
            thrust::device_vector<SuperclusterInfo>(numISuperclusters)};

        if (numISuperclusters == 0) return nbList;

        thrust::device_vector<typename Config::SuperclusterSplitMask> superclusterSplitMasks(numISuperclusters);
        {
            constexpr unsigned numThreads = 128;
            const unsigned numBlocks      = iceil(numISuperclusters, numThreads);
            initSuperclusterInfo<<<numBlocks, numThreads>>>(firstISupercluster, lastISupercluster,
                                                            rawPtr(nbList.superclusterInfo));
            checkGpuErrors(cudaGetLastError());
        }
        {
            constexpr unsigned numThreads = 128;
            const unsigned numBlocks      = iceil(groups.numGroups, numThreads);
            computeSuperclusterSplitMasks<Config><<<numBlocks, numThreads>>>(
                firstISupercluster, lastISupercluster, firstValidBody, groups, rawPtr(superclusterSplitMasks));
            checkGpuErrors(cudaGetLastError());
        }

        thrust::device_vector<Vec3<Tc>> jClusterBboxCenters(numJClusters), jClusterBboxSizes(numJClusters);

        {
            constexpr unsigned numThreads = 128;
            unsigned numBlocks            = iceil(numJClusters * Config::jSize, numThreads);
            computeJClusterBboxes<Config><<<numBlocks, numThreads>>>(
                firstValidBody, totalBodies, x, y, z, rawPtr(jClusterBboxCenters), rawPtr(jClusterBboxSizes));
            checkGpuErrors(cudaGetLastError());
        }

        thrust::device_vector<GlobalBuildData> globalBuildData(1);

        constexpr unsigned numSuperclustersPerBlock =
            64 / (Config::iThreads * Config::jSize / Config::numWarpsPerInteraction);
        const dim3 blockSize = {Config::iThreads, Config::jSize / Config::numWarpsPerInteraction,
                                numSuperclustersPerBlock};
        const unsigned numBlocks =
            std::min(GpuConfig::smCount * (TravConfig::numWarpsPerSm / numSuperclustersPerBlock),
                     (numISuperclusters + numSuperclustersPerBlock - 1) / numSuperclustersPerBlock);

        thrust::device_vector<int> globalPool(TravConfig::memPerWarp * numSuperclustersPerBlock * numBlocks);
        Th maxH = 0;
        if constexpr (Config::symmetric)
            maxH = thrust::reduce(thrust::device, h + firstValidBody, h + totalBodies, Th(0), thrust::maximum<Th>());

        const auto runBuildKernel = [&]
        {
            checkGpuErrors(cudaMemsetAsync(rawPtr(globalBuildData), 0, sizeof(GlobalBuildData)));

            auto run = [&](auto usePbc)
            {
                buildNbList<Config, numSuperclustersPerBlock, decltype(usePbc)::value><<<numBlocks, blockSize>>>(
                    tree, box, firstValidBody, totalBodies, groups.firstBody, groups.lastBody, x, y, z, h, maxH,
                    rawPtr(jClusterBboxCenters), rawPtr(jClusterBboxSizes), rawPtr(superclusterSplitMasks),
                    rawPtr(nbList.neighborData), nbList.neighborData.size(), rawPtr(nbList.superclusterInfo),
                    nbList.superclusterInfo.size(), rawPtr(globalPool), rawPtr(globalBuildData));
            };
            if (box.boundaryX() == BoundaryType::periodic | box.boundaryY() == BoundaryType::periodic |
                box.boundaryZ() == BoundaryType::periodic)
                run(std::true_type());
            else
                run(std::false_type());
            checkGpuErrors(cudaGetLastError());
        };

        runBuildKernel();

        unsigned long long requiredSize;
        checkGpuErrors(cudaMemcpy(&requiredSize, &rawPtr(globalBuildData)->neighborDataSize, sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost));
        if (requiredSize > nbList.neighborData.size())
        {
            nbList.neighborData.resize(requiredSize);
            runBuildKernel();
            checkGpuErrors(cudaDeviceSynchronize());
        }

        thrust::stable_sort(thrust::device, nbList.superclusterInfo.begin(), nbList.superclusterInfo.end());

        return nbList;
    }
};

} // namespace cstone::ijloop
