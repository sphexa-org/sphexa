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

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include "cstone/compressneighbors.cuh"
#include "cstone/cuda/cub.hpp"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/reducearray.cuh"
#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/traversal/ijloop/atomic_update_ptr.cuh"
#include "cstone/traversal/ijloop/ijloop.hpp"
#include "cstone/util/uninitialized.hpp"
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

__global__ static void initSuperclusterInfo(const LocalIndex firstISupercluster,
                                            const LocalIndex lastISupercluster,
                                            SuperclusterInfo* superclusterInfo)
{
    const LocalIndex index = blockIdx.x * blockDim.x + threadIdx.x;

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
    const LocalIndex index = blockIdx.x * blockDim.x + threadIdx.x;
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
        newSplitMask = oldSplitMask | ((typename Config::SuperclusterSplitMask)(1) << splitPosition);
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

    const unsigned laneIdx = laneIndex();

    const unsigned i = threadIdx.x + blockDim.x * blockIdx.x;

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

        const unsigned idx = laneIdx % Config::jSize;
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
            bboxMin = {std::min(shflDownSync(bboxMin[0], offset), bboxMin[0]),
                       std::min(shflDownSync(bboxMin[1], offset), bboxMin[1]),
                       std::min(shflDownSync(bboxMin[2], offset), bboxMin[2])};
            bboxMax = {std::max(shflDownSync(bboxMax[0], offset), bboxMax[0]),
                       std::max(shflDownSync(bboxMax[1], offset), bboxMax[1]),
                       std::max(shflDownSync(bboxMax[2], offset), bboxMax[2])};
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

template<class Config, unsigned NumSuperclustersPerBlock>
__device__ inline void sortCandidates(std::uint32_t* candidates, unsigned numCandidates)
{
    const unsigned laneIdx = laneIndex();
    assert(blockDim.x * blockDim.y == GpuConfig::warpSize);
    assert(blockDim.z == NumSuperclustersPerBlock);

    constexpr unsigned itemsPerWarp = Config::ncMax / GpuConfig::warpSize;
    std::uint32_t items[itemsPerWarp];
#pragma unroll
    for (unsigned i = 0; i < itemsPerWarp; ++i)
    {
        const unsigned c = laneIdx * itemsPerWarp + i;
        items[i]         = c < numCandidates ? candidates[c] : std::numeric_limits<std::uint32_t>::max();
    }

    using WarpSort = cub::WarpMergeSort<std::uint32_t, itemsPerWarp, GpuConfig::warpSize>;
    __shared__ typename WarpSort::TempStorage sortTmp[NumSuperclustersPerBlock];
    WarpSort(sortTmp[threadIdx.z]).Sort(items, std::less<unsigned>());

#pragma unroll
    for (unsigned i = 0; i < itemsPerWarp; ++i)
    {
        const unsigned c = laneIdx * itemsPerWarp + i;
        if (c < numCandidates) candidates[c] = items[i];
    }

    syncWarp();
}

template<class Config, unsigned NumSuperclustersPerBlock, bool UsePbc, class Tc, class Th>
__device__ inline void pruneCandidates(const Box<Tc>& box,
                                       const LocalIndex firstValidBody,
                                       const LocalIndex totalBodies,
                                       const Tc* const __restrict__ x,
                                       const Tc* const __restrict__ y,
                                       const Tc* const __restrict__ z,
                                       const Th* const __restrict__ h,
                                       const Th searchExtFactor,
                                       const unsigned iSupercluster,
                                       std::uint32_t* __restrict__ jClusters,
                                       unsigned& numCandidates)
{
    const unsigned laneIdx = laneIndex();
    assert(blockDim.x * blockDim.y == GpuConfig::warpSize);
    assert(blockDim.z == NumSuperclustersPerBlock);

    __shared__ Tc xisBuffer[NumSuperclustersPerBlock][Config::superclusterSize];
    __shared__ Tc yisBuffer[NumSuperclustersPerBlock][Config::superclusterSize];
    __shared__ Tc zisBuffer[NumSuperclustersPerBlock][Config::superclusterSize];
    __shared__ Th hisBuffer[NumSuperclustersPerBlock][Config::superclusterSize];
    Tc* xis = xisBuffer[threadIdx.z];
    Tc* yis = yisBuffer[threadIdx.z];
    Tc* zis = zisBuffer[threadIdx.z];
    Th* his = hisBuffer[threadIdx.z];

    for (unsigned n = laneIdx; n < Config::superclusterSize; n += GpuConfig::warpSize)
    {
        const unsigned i =
            std::max(std::min(Config::superclusterSize * iSupercluster + n, totalBodies - 1), firstValidBody);
        xis[n] = x[i];
        yis[n] = y[i];
        zis[n] = z[i];
        his[n] = h[i];
    }

    syncWarp();

    constexpr unsigned iClustersPerWarp = Config::iThreads / Config::iSize;
    const unsigned iClusterOffset       = iClustersPerWarp == 1 ? 0 : threadIdx.x / Config::iSize;

    std::uint32_t previousJCluster = std::numeric_limits<std::uint32_t>::max();
    unsigned numJClusters          = 0;
    for (unsigned candidate = 0; candidate < numCandidates; ++candidate)
    {
        const std::uint32_t jCluster = jClusters[candidate];
        if (jCluster == previousJCluster) continue;
        previousJCluster = jCluster;

        bool keep = false;
        for (unsigned w = 0; w < Config::numWarpsPerInteraction; ++w)
        {
            const unsigned j =
                jCluster * Config::jSize + (Config::jSize / Config::numWarpsPerInteraction) * w + threadIdx.y;
            const unsigned jSupercluster = superclusterIndex<Config>(j);
            const unsigned jClamped      = std::max(firstValidBody, std::min(j, totalBodies - 1));
            const Tc xj                  = x[jClamped];
            const Tc yj                  = y[jClamped];
            const Tc zj                  = z[jClamped];
            const Th hj                  = h[jClamped];

            for (unsigned c = 0; c < Config::iClustersPerSupercluster; c += iClustersPerWarp)
            {
                const unsigned ci = c + iClusterOffset;
                const unsigned i  = ci * Config::iSize + threadIdx.x % Config::iSize;
                if (!Config::symmetric | (iSupercluster != jSupercluster) | (i <= j))
                {
                    const unsigned si = ci * Config::iSize + threadIdx.x % Config::iSize;
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
                    const Th distSq = xij * xij + yij * yij + zij * zij;
                    const Th hMax   = (Config::symmetric ? std::max(hi, hj) : hi) * searchExtFactor;
                    keep            = distSq < Th(4) * hMax * hMax;
                }
                keep = anySync(keep);
                if (keep) break;
            }
            if (keep) break;
        }
        if (keep)
        {
            if (laneIdx == 0) jClusters[numJClusters] = jCluster;
            ++numJClusters;
        }
    }
    numCandidates = numJClusters;
    syncWarp();
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
    const unsigned laneIdx = laneIndex();
    assert(blockDim.x * blockDim.y == GpuConfig::warpSize);
    assert(blockDim.z == NumSuperclustersPerBlock);

    Vec3<Tc> bbMin = {std::numeric_limits<Tc>::max(), std::numeric_limits<Tc>::max(), std::numeric_limits<Tc>::max()};
    Vec3<Tc> bbMax = {std::numeric_limits<Tc>::lowest(), std::numeric_limits<Tc>::lowest(),
                      std::numeric_limits<Tc>::lowest()};
    assert(lastGroupParticle - firstGroupParticle > 0);

    for (unsigned i = firstGroupParticle + laneIdx; i < lastGroupParticle; i += GpuConfig::warpSize)
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
        bbMin[d] = warpMin(bbMin[d]);
        bbMax[d] = warpMax(bbMax[d]);
    }

    const Vec3<Tc> groupCenter = (bbMax + bbMin) * Tc(0.5);
    const Vec3<Tc> groupSize   = (bbMax - bbMin) * Tc(0.5) * tree.searchExtFactor;

    const bool usePbc = UsePbc && !insideBox(groupCenter, groupSize, box);

    const unsigned firstISupercluster = superclusterIndex<Config>(firstBody);
    const unsigned lastISupercluster  = superclusterIndex<Config>(lastBody - 1) + 1;
    const unsigned iSupercluster      = superclusterIndex<Config>(firstGroupParticle);
    const unsigned numJClusters       = jClusterIndex<Config>(totalBodies - 1) + 1;

    const auto checkOverlap = [&](const unsigned jCluster, const unsigned numLanesValid)
    {
        assert(numLanesValid > 0);

        const unsigned prevJCluster = shflUpSync(jCluster, 1);
        bool isNeighbor = laneIdx < numLanesValid & jCluster < numJClusters & (laneIdx == 0 | prevJCluster != jCluster);

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

        const unsigned nbIndex    = exclusiveScanBool(isNeighbor);
        unsigned newNumCandidates = shflSync(numCandidates + nbIndex + isNeighbor, GpuConfig::warpSize - 1);
        if (newNumCandidates >= Config::ncMax)
        {
            sortCandidates<Config, NumSuperclustersPerBlock>(candidates, numCandidates);
            pruneCandidates<Config, NumSuperclustersPerBlock, UsePbc>(box, firstValidBody, totalBodies, x, y, z, h,
                                                                      (Th)tree.searchExtFactor, iSupercluster,
                                                                      candidates, numCandidates);
            newNumCandidates = shflSync(numCandidates + nbIndex + isNeighbor, GpuConfig::warpSize - 1);
        }
        // TODO: proper error handling
        assert(newNumCandidates < Config::ncMax);
        if (isNeighbor & (numCandidates + nbIndex < Config::ncMax)) candidates[numCandidates + nbIndex] = jCluster;
        numCandidates = newNumCandidates;
    };

    volatile __shared__ int sharedPool[NumSuperclustersPerBlock * GpuConfig::warpSize];

    int jClusterQueue; // warp queue for source jCluster indices
    volatile int* tempQueue = sharedPool + GpuConfig::warpSize * threadIdx.z;
    int* cellQueue = globalPool + TravConfig::memPerWarp * (blockIdx.x * NumSuperclustersPerBlock + threadIdx.z);
    const TreeNodeIndex* __restrict__ childOffsets   = tree.childOffsets;
    const TreeNodeIndex* __restrict__ internalToLeaf = tree.internalToLeaf;
    const LocalIndex* __restrict__ layout            = tree.layout;
    const Vec3<Tc>* __restrict__ centers             = tree.centers;
    const Vec3<Tc>* __restrict__ sizes               = tree.sizes;

    // populate initial cell queue
    if (laneIdx == 0) cellQueue[0] = 1;
    syncWarp();

    // these variables are always identical on all warp lanes
    int numSources        = 1; // current stack size
    int newSources        = 0; // stack size for next level
    int oldSources        = 0; // cell indices done
    int sourceOffset      = 0; // current level stack pointer, once this reaches numSources, the level is done
    int jClusterFillLevel = 0; // fill level of the source jCluster warp queue

    while (numSources > 0) // While there are source cells to traverse
    {
        int sourceIdx   = sourceOffset + laneIdx;
        int sourceQueue = 0;
        if (laneIdx < GpuConfig::warpSize / 8)
            sourceQueue = cellQueue[ringAddr(oldSources + sourceIdx)]; // Global source cell index in queue
        sourceQueue         = spreadSeg8(sourceQueue);
        sourceIdx           = shflSync(sourceIdx, laneIdx / 8);
        const bool isSource = sourceIdx < numSources; // Source index is within bounds
        if (!isSource) sourceQueue = 0;

        const Vec3<Tc> curSrcCenter = centers[sourceQueue];      // Current source cell center
        const Vec3<Tc> curSrcSize   = sizes[sourceQueue];        // Current source cell center
        const int childBegin        = childOffsets[sourceQueue]; // First child cell
        const bool isNode           = childBegin;
        const bool isClose          = usePbc ? cellOverlap<true>(curSrcCenter, curSrcSize, groupCenter, groupSize, box)
                                             : cellOverlap<false>(curSrcCenter, curSrcSize, groupCenter, groupSize, box);
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
        const int numJClusters =
            (layout[leafIdx + 1] == layout[leafIdx]
                 ? 0
                 : jClusterIndex<Config>(layout[leafIdx + 1] + firstValidBody - 1) + 1 - firstJCluster) &
            -int(isDirect); // Number of jClusters in cell
        bool directTodo            = numJClusters;
        const int numJClustersScan = inclusiveScanInt(numJClusters);  // Inclusive scan of numJClusters
        int numJClustersLane       = numJClustersScan - numJClusters; // Exclusive scan of numJClusters
        int numJClustersWarp =
            shflSync(numJClustersScan, GpuConfig::warpSize - 1); // Total numJClusters of current warp
        int prevJClusterIdx = 0;
        while (numJClustersWarp > 0) // While there are jClusters to process from current source cell set
        {
            tempQueue[laneIdx] = 1; // Default scan input is 1, such that consecutive lanes load consecutive bodies
            if (directTodo && (numJClustersLane < GpuConfig::warpSize))
            {
                directTodo                  = false;              // Set cell as processed
                tempQueue[numJClustersLane] = -1 - firstJCluster; // Put first source cell body index into the queue
            }
            const int jClusterIdx = inclusiveSegscanInt(tempQueue[laneIdx], prevJClusterIdx);
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
                jClusterQueue = (laneIdx < jClusterFillLevel) ? jClusterQueue : topUp;

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

template<class Config, unsigned NumSuperclustersPerBlock, bool UsePbc, class Tc, class Th>
__device__ __forceinline__ void pruneCandidatesAndComputeMasks(const Box<Tc>& box,
                                                               const LocalIndex firstValidBody,
                                                               const LocalIndex totalBodies,
                                                               const Tc* const __restrict__ x,
                                                               const Tc* const __restrict__ y,
                                                               const Tc* const __restrict__ z,
                                                               const Th* const __restrict__ h,
                                                               const Th searchExtFactor,
                                                               const unsigned iSupercluster,
                                                               std::uint32_t* __restrict__ jClusters,
                                                               std::uint32_t* __restrict__ masks,
                                                               const unsigned numCandidates,
                                                               unsigned& numJClusters)
{
    const unsigned laneIdx = laneIndex();
    assert(blockDim.x * blockDim.y == GpuConfig::warpSize);
    assert(blockDim.z == NumSuperclustersPerBlock);

    __shared__ Tc xisBuffer[NumSuperclustersPerBlock][Config::superclusterSize];
    __shared__ Tc yisBuffer[NumSuperclustersPerBlock][Config::superclusterSize];
    __shared__ Tc zisBuffer[NumSuperclustersPerBlock][Config::superclusterSize];
    __shared__ Th hisBuffer[NumSuperclustersPerBlock][Config::superclusterSize];
    Tc* xis = xisBuffer[threadIdx.z];
    Tc* yis = yisBuffer[threadIdx.z];
    Tc* zis = zisBuffer[threadIdx.z];
    Th* his = hisBuffer[threadIdx.z];

    for (unsigned n = laneIdx; n < Config::superclusterSize; n += GpuConfig::warpSize)
    {
        const unsigned i =
            std::max(std::min(Config::superclusterSize * iSupercluster + n, totalBodies - 1), firstValidBody);
        xis[n] = x[i];
        yis[n] = y[i];
        zis[n] = z[i];
        his[n] = h[i];
    }

    const unsigned maxMasksSize = masksSize<Config>(numCandidates);
    for (unsigned n = laneIdx; n < maxMasksSize; n += GpuConfig::warpSize)
        masks[n] = 0;

    syncWarp();

    constexpr unsigned iClustersPerWarp = Config::iThreads / Config::iSize;
    const unsigned iClusterOffset       = iClustersPerWarp == 1 ? 0 : threadIdx.x / Config::iSize;

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
            const unsigned j =
                jCluster * Config::jSize + (Config::jSize / Config::numWarpsPerInteraction) * w + threadIdx.y;
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
                    const unsigned i  = ci * Config::iSize + threadIdx.x % Config::iSize;
                    if (!Config::symmetric | (iSupercluster != jSupercluster) | (i <= j))
                    {
                        const unsigned si = ci * Config::iSize + threadIdx.x % Config::iSize;
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
                        const Th hMax               = (Config::symmetric ? std::max(hi, hj) : hi) * searchExtFactor;
                        const bool overlaps         = distSq < Th(4) * hMax * hMax;
                        const unsigned maskBitIndex = w * Config::iClustersPerSupercluster + ci;
                        assert(maskBitIndex < 32);
                        mask |= std::uint32_t(overlaps) << maskBitIndex;
                    }
                }
            }
        }
        mask = warpBitwiseOr(mask);
        if (mask)
        {
            if (laneIdx == 0)
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
    const unsigned laneIdx = laneIndex();
    assert(blockDim.x * blockDim.y == GpuConfig::warpSize);
    assert(blockDim.z == NumSuperclustersPerBlock);

    const unsigned mSize = masksSize<Config>(info.neighborsCount);
    unsigned nbSize      = info.neighborsCount;

    __shared__ std::uint32_t compressedJClusters[NumSuperclustersPerBlock][Config::compress ? Config::ncMax : 1];

    if constexpr (Config::compress)
    {
        warpCompressNeighbors(jClusters, (char*)compressedJClusters[threadIdx.z], info.neighborsCount);
        nbSize = compressedNeighborsSize((const char*)compressedJClusters[threadIdx.z]);
    }

    const unsigned long long totalSize = nbSize + mSize;
    if (laneIdx == 0) info.dataIndex = atomicAdd(&globalBuildData->neighborDataSize, totalSize);
    info.dataIndex = shflSync(info.dataIndex, 0);

    for (unsigned n = laneIdx; n < mSize; n += GpuConfig::warpSize)
    {
        const auto index = info.dataIndex + n;
        if (index < neighborDataSize) neighborData[index] = masks[n];
    }

    for (unsigned n = laneIdx; n < nbSize; n += GpuConfig::warpSize)
    {
        const auto index = info.dataIndex + mSize + n;
        if (index < neighborDataSize)
            neighborData[index] = (Config::compress ? compressedJClusters[threadIdx.z] : jClusters)[n];
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
    const unsigned laneIdx = laneIndex();
    assert(blockDim.x == Config::iThreads);
    assert(blockDim.y == GpuConfig::warpSize / Config::iThreads);
    assert(blockDim.z == NumSuperclustersPerBlock);

    while (true)
    {
        unsigned index;
        if (laneIdx == 0) index = atomicAdd(&globalBuildData->index, 1);
        index = shflSync(index, 0);
        if (index >= numSuperClusters) return;

        SuperclusterInfo info = {.index = superclusterInfo[index].index, .neighborsCount = 0, .dataIndex = 0};

        __shared__ std::uint32_t jClustersBuffer[NumSuperclustersPerBlock][Config::ncMax];
        std::uint32_t* jClusters = jClustersBuffer[threadIdx.z];

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
        }

        __shared__ std::uint32_t masksBuffer[NumSuperclustersPerBlock][masksSize<Config>(Config::ncMax)];
        std::uint32_t* masks = masksBuffer[threadIdx.z];

        sortCandidates<Config, NumSuperclustersPerBlock>(jClusters, numCandidates);
        pruneCandidatesAndComputeMasks<Config, NumSuperclustersPerBlock, UsePbc>(
            box, firstValidBody, totalBodies, x, y, z, h, (Th)tree.searchExtFactor, info.index, jClusters, masks,
            numCandidates, info.neighborsCount);

        storeNeighborData<Config, NumSuperclustersPerBlock>(jClusters, masks, neighborData, neighborDataSize, info,
                                                            globalBuildData);

        if (laneIdx == 0) superclusterInfo[index] = info;
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

template<class Config, class T0, class... T, class... Ps, class Postamble, class IData>
__device__ __forceinline__ void storeTupleISum(std::tuple<T0, T...> tuple,
                                               std::tuple<Ps*...> const& ptrs,
                                               const unsigned index,
                                               const bool store,
                                               Postamble const& postamble,
                                               IData const& iData)
{
    assert(blockDim.x == Config::iThreads);

    if constexpr (std::conjunction_v<std::is_same<T0, T>...> && sizeof...(T) < GpuConfig::warpSize / Config::iThreads &&
                  std::is_same<Postamble, detail::EmptyPostamble>())
    {
        const T0 res =
            reduceTuple<GpuConfig::warpSize / Config::iThreads, true>(tuple,
                                                                      [](auto result, auto const& value)
                                                                      {
                                                                          detail::updateResultImpl(result, value);
                                                                          return result;
                                                                      });
        if ((threadIdx.y % (GpuConfig::warpSize / Config::iThreads) <= sizeof...(T)) & store)
        {
            auto* ptr = dynamicTupleGet(ptrs, threadIdx.y % (GpuConfig::warpSize / Config::iThreads));
            if constexpr (Config::symmetric | (Config::numWarpsPerInteraction > 1))
                atomicUpdatePtr(&ptr[index], res);
            else
                ptr[index] = detail::unwrapModifiersImpl(res);
        }
    }
    else
    {
#pragma unroll
        for (unsigned offset = GpuConfig::warpSize / 2; offset >= Config::iThreads; offset /= 2)
            util::for_each_tuple([&](auto& t) { detail::updateResultImpl(t, shflDownSync(t, offset)); }, tuple);

        if ((threadIdx.y % (GpuConfig::warpSize / Config::iThreads) == 0) & store)
        {
            if constexpr (Config::symmetric | Config::numWarpsPerInteraction > 1)
            {
                util::for_each_tuple([index](auto* ptr, auto const& t) { atomicUpdatePtr(&ptr[index], t); }, ptrs,
                                     tuple);
            }
            else { storeParticleData(ptrs, index, postamble(iData, unwrapModifiers(tuple))); }
        }
    }
}

template<class Config, class T0, class... T, class... Ps>
constexpr __device__ void
storeTupleJSum(std::tuple<T0, T...> tuple, std::tuple<Ps*...> const& ptrs, const unsigned index, const bool store)
{
    assert(blockDim.x == Config::iThreads);

    if constexpr (std::conjunction_v<std::is_same<T0, T>...> && sizeof...(T) < Config::iThreads)
    {
        const T0 res = reduceTuple<Config::iThreads, false>(tuple,
                                                            [](auto result, auto const& value)
                                                            {
                                                                detail::updateResultImpl(result, value);
                                                                return result;
                                                            });
        if ((threadIdx.x <= sizeof...(T)) & store)
        {
            auto* ptr = dynamicTupleGet(ptrs, threadIdx.x);
            atomicUpdatePtr(&ptr[index], res);
        }
    }
    else
    {
#pragma unroll
        for (unsigned offset = Config::iThreads / 2; offset >= 1; offset /= 2)
            util::for_each_tuple([&](auto& t) { detail::updateResultImpl(t, shflDownSync(t, offset)); }, tuple);

        if ((threadIdx.x == 0) & store)
            util::for_each_tuple([index](auto* ptr, auto const& t) { atomicUpdatePtr(&ptr[index], t); }, ptrs, tuple);
    }
}

template<std::size_t Size, class... Ts>
constexpr std::tuple<std::array<Ts, Size>...> buffersForResults(std::tuple<Ts...> const&)
{
    return {};
}

template<class Config,
         unsigned NumSuperclustersPerBlock,
         bool UsePbc,
         class Tc,
         class Th,
         class In,
         class Out,
         class Interaction,
         class Postamble>
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
    const Postamble postamble,
    const std::uint32_t* const __restrict__ neighborData,
    const SuperclusterInfo* const __restrict__ superclusterInfo)
{
    static_assert(Config::ncMax % GpuConfig::warpSize == 0);
    static_assert(NumSuperclustersPerBlock > 0);
    static_assert(Config::iThreads * Config::jSize >= GpuConfig::warpSize);
    static_assert(Config::iThreads * Config::jSize % GpuConfig::warpSize == 0);

    assert(blockDim.x == Config::iThreads);
    assert(blockDim.y == Config::jSize);
    assert(blockDim.z == NumSuperclustersPerBlock);

    const unsigned firstISupercluster = superclusterIndex<Config>(firstBody);
    const unsigned lastISupercluster  = superclusterIndex<Config>(lastBody - 1) + 1;
    const unsigned numISuperclusters  = lastISupercluster - firstISupercluster;
    const unsigned iSuperclusterIndex = blockIdx.x * NumSuperclustersPerBlock + threadIdx.z;
    if (iSuperclusterIndex >= numISuperclusters) return;

    auto [iSupercluster, iSuperclusterNeighborsCount, iSuperclusterDataIndex] = superclusterInfo[iSuperclusterIndex];

    using particleData_t = decltype(loadParticleData(x, y, z, h, input, 0));

    // TODO: bank-conflict friendly SoA layout?
    __shared__ util::Uninitialized<particleData_t[NumSuperclustersPerBlock][Config::iClustersPerSupercluster * Config::iSize]> iSuperclusterDataBuffer;
    particleData_t* iSuperclusterData = iSuperclusterDataBuffer.data()[threadIdx.z];
    {
        const unsigned base = iSupercluster * Config::superclusterSize;
        for (unsigned offset = threadIdx.y * Config::iThreads + threadIdx.x; offset < Config::superclusterSize;
             offset += Config::iThreads * Config::jSize)
        {
            const unsigned i = base + offset;
            auto iData       = (i >= firstValidBody & i < totalBodies) ? loadParticleData(x, y, z, h, input, i)
                                                                       : dummyParticleData(x, y, z, h, input, i);
            std::get<0>(iData) -= firstValidBody;
            iSuperclusterData[offset] = iData;
        }
    }

    __shared__ unsigned nbDataBuffer[NumSuperclustersPerBlock][Config::ncMax + masksSize<Config>(Config::ncMax)];
    unsigned* const nbData = nbDataBuffer[threadIdx.z];

    const unsigned maskSize   = masksSize<Config>(iSuperclusterNeighborsCount);
    const unsigned nbDataSize = iSuperclusterNeighborsCount + maskSize;

    constexpr unsigned iClustersPerWarp = Config::iThreads / Config::iSize;
    const unsigned warpIndex            = threadIdx.y / (Config::jSize / Config::numWarpsPerInteraction);

    if constexpr (Config::compress)
    {
        for (unsigned n = threadIdx.y * Config::iThreads + threadIdx.x; n < maskSize;
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
        for (unsigned n = threadIdx.y * Config::iThreads + threadIdx.x; n < nbDataSize;
             n += Config::iThreads * Config::jSize)
            nbData[n] = neighborData[iSuperclusterDataIndex + n];
    }

    __syncthreads();

    using result_t = std::decay_t<decltype(interaction(particleData_t(), particleData_t(), Vec3<Tc>(), Tc(0)))>;
    static_assert(
        !Config::symmetric || std::is_same<std::decay_t<decltype(postamble(particleData_t(),
                                                                           unwrapModifiers(result_t())))>,
                                           decltype(unwrapModifiers(result_t()))>(),
        "postamble that changes the result type is not supported in combination with symmetric neighborhood or more "
        "than one warp per cluster-cluster interaction");

    std::array<result_t, Config::iClustersPerSupercluster / iClustersPerWarp> iResults = {};
    const unsigned iClusterOffset = iClustersPerWarp == 1 ? 0 : threadIdx.x / Config::iSize;

    for (unsigned nb = 0; nb < iSuperclusterNeighborsCount; ++nb)
    {
        const unsigned maskStartIndex = nb * (Config::iClustersPerSupercluster * Config::numWarpsPerInteraction) +
                                        (warpIndex * Config::iClustersPerSupercluster);
        unsigned warpMask =
            (nbData[maskStartIndex / 32] >> (maskStartIndex % 32)) & ((1 << Config::iClustersPerSupercluster) - 1);

        if (warpMask)
        {
            const unsigned jCluster      = nb < iSuperclusterNeighborsCount ? nbData[nb + maskSize] : ~0u;
            const unsigned j             = jCluster * Config::jSize + threadIdx.y;
            const unsigned jSupercluster = superclusterIndex<Config>(j);
            auto jData                   = (nb < iSuperclusterNeighborsCount & j >= firstValidBody & j < totalBodies)
                                               ? loadParticleData(x, y, z, h, input, j)
                                               : dummyParticleData(x, y, z, h, input, j);
            std::get<0>(jData) -= firstValidBody;
            result_t jResult = {};

            warpMask >>= iClusterOffset;
            unsigned i = iSupercluster * Config::superclusterSize + threadIdx.x;
            for (unsigned c = 0; c < Config::iClustersPerSupercluster; c += iClustersPerWarp)
            {
                if ((warpMask & 1) && (!Config::symmetric | (iSupercluster != jSupercluster) | (i <= j)))
                {
                    const auto& iData = iSuperclusterData[c * Config::iSize + threadIdx.x];
                    assert(std::get<0>(iData) == i - firstValidBody);
                    const auto [ijPosDiff, distSq] = posDiffAndDistSq(UsePbc, box, iData, jData);
                    const auto ijInteraction       = interaction(iData, jData, ijPosDiff, distSq);
                    if (distSq < radiusSq(iData)) updateResult(iResults[c / iClustersPerWarp], ijInteraction);
                    if constexpr (Config::symmetric)
                    {
                        if ((distSq < radiusSq(jData)) & ((i != j) | (i < firstBody) | (i >= lastBody)))
                        {
                            const auto jiInteraction =
                                selectSymmetric(ijInteraction, interaction(jData, iData, -ijPosDiff, distSq));
                            updateResult(jResult, jiInteraction);
                        }
                    }
                }
                warpMask >>= iClustersPerWarp;
                i += Config::iThreads;
            }

            if constexpr (Config::symmetric)
            {
                storeTupleJSum<Config>(jResult, output, j, j >= firstBody & j < lastBody);
            }
        }
    }

    if constexpr (!Config::symmetric && Config::numWarpsPerInteraction > 1)
    {
        __shared__ decltype(buffersForResults<Config::superclusterSize>(
            unwrapModifiers(result_t()))) outputBuffers[NumSuperclustersPerBlock];
        auto outputBufferPtrs = util::tupleMap([](auto& array) { return array.data(); }, outputBuffers[threadIdx.z]);
        auto init             = unwrapModifiers(result_t{});
        for (unsigned offset = threadIdx.y * Config::iThreads + threadIdx.x; offset < Config::superclusterSize;
             offset += Config::iThreads * Config::jSize)
            storeParticleData(outputBufferPtrs, offset, init);

        __syncthreads();

        for (unsigned c = 0; c < Config::iClustersPerSupercluster; c += iClustersPerWarp)
        {
            storeTupleISum<Config>(iResults[c / iClustersPerWarp], outputBufferPtrs, c * Config::iSize + threadIdx.x,
                                   true, detail::EmptyPostamble{}, iSuperclusterData[c * Config::iSize + threadIdx.x]);
        }

        __syncthreads();

        const unsigned base = iSupercluster * Config::superclusterSize;
        for (unsigned offset = threadIdx.y * Config::iThreads + threadIdx.x; offset < Config::superclusterSize;
             offset += Config::iThreads * Config::jSize)
        {
            const unsigned i = base + offset;
            if (i >= firstBody & i < lastBody)
            {
                const auto iData   = iSuperclusterData[offset];
                const auto iResult = util::tupleMap([&](auto const* ptr) { return ptr[offset]; }, outputBufferPtrs);
                storeParticleData(output, i, postamble(iData, unwrapModifiers(iResult)));
            }
        }
    }
    else
    {
        for (unsigned c = 0; c < Config::iClustersPerSupercluster; c += iClustersPerWarp)
        {
            const auto i = iSupercluster * Config::superclusterSize + c * Config::iSize + threadIdx.x;
            storeTupleISum<Config>(iResults[c / iClustersPerWarp], output, i, i >= firstBody & i < lastBody, postamble,
                                   iSuperclusterData[c * Config::iSize + threadIdx.x]);
        }
    }
}

template<class Tc, class Th, class In, class Out, class Interaction>
__global__ void initResult(const LocalIndex firstBody,
                           const LocalIndex lastBody,
                           const Tc* __restrict__ x,
                           const Tc* __restrict__ y,
                           const Tc* __restrict__ z,
                           const Th* __restrict__ h,
                           const In __grid_constant__ input,
                           const Out __grid_constant__ output,
                           Interaction interaction)
{
    const LocalIndex i = blockDim.x * blockIdx.x + threadIdx.x + firstBody;
    if (i >= lastBody) return;

    using IData  = decltype(loadParticleData(x, y, z, h, input, 0));
    using Result = decltype(interaction(IData{}, IData{}, Vec3<Tc>{0, 0, 0}, Tc(0)));
    storeParticleData(output, i, unwrapModifiers(Result{}));
}

template<class Tc, class Th, class In, class Out, class Postamble>
__global__ void applyPostamble(const LocalIndex totalBodies,
                               const Tc* __restrict__ x,
                               const Tc* __restrict__ y,
                               const Tc* __restrict__ z,
                               const Th* __restrict__ h,
                               const In __grid_constant__ input,
                               const Out __grid_constant__ output,
                               const Postamble postamble)
{
    const LocalIndex i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > totalBodies) return;

    const auto iData  = loadParticleData(x, y, z, h, input, i);
    const auto result = util::tupleMap([&](auto* ptr) { return ptr[i]; }, output);
    storeParticleData(output, i, postamble(iData, result));
}

template<class Config, class Tc, class Th>
struct GpuSuperclusterNbListNeighborhoodImpl
{
    Box<Tc> box = {0, 0};
    LocalIndex firstValidBody, totalBodies, firstBody, lastBody;
    const Tc *x, *y, *z;
    const Th* h;
    thrust::device_vector<std::uint32_t> neighborData;
    thrust::device_vector<SuperclusterInfo> superclusterInfo;

    template<class... In, class... Out, class Interaction, class Postamble>
    void
    ijLoop(std::tuple<In*...> input, std::tuple<Out*...> output, Interaction&& interaction, Postamble&& postamble) const
    {
        const LocalIndex numBodies = lastBody - firstBody;
        if (numBodies == 0) return;

        util::for_each_tuple([&](auto& ptr) { ptr -= firstValidBody; }, input);
        util::for_each_tuple([&](auto& ptr) { ptr -= firstValidBody; }, output);

        if constexpr (Config::symmetric)
        {
            constexpr unsigned threads = 256;
            const unsigned numBlocks   = iceil(numBodies, threads);
            initResult<<<numBlocks, threads>>>(firstBody, lastBody, x, y, z, h, makeConstRestrict(input), output,
                                               std::forward<Interaction>(interaction));
            checkGpuErrors(cudaGetLastError());
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
            runIjLoop<Config, numSuperclustersPerBlock, decltype(usePbc)::value><<<numBlocks, blockSize>>>(
                box, firstValidBody, totalBodies, firstBody, lastBody, x, y, z, h, makeConstRestrict(input), output,
                std::forward<Interaction>(interaction), std::forward<Postamble>(postamble), rawPtr(neighborData),
                rawPtr(superclusterInfo));
            checkGpuErrors(cudaGetLastError());
        };
        if (box.boundaryX() == BoundaryType::periodic | box.boundaryY() == BoundaryType::periodic |
            box.boundaryZ() == BoundaryType::periodic)
            runKernel(std::true_type());
        else
            runKernel(std::false_type());

        if constexpr (Config::symmetric && !std::is_same<std::decay_t<Postamble>, detail::EmptyPostamble>())
        {
            util::for_each_tuple([&](auto& ptr) { ptr += firstValidBody; }, input);
            util::for_each_tuple([&](auto& ptr) { ptr += firstValidBody; }, output);
            constexpr unsigned threads = 256;
            const unsigned numBlocks   = iceil(totalBodies, threads);
            applyPostamble<<<numBlocks, threads>>>(totalBodies - firstValidBody, x + firstValidBody, y + firstValidBody,
                                                   z + firstValidBody, h + firstValidBody, makeConstRestrict(input),
                                                   output, std::forward<Postamble>(postamble));
        }
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
    template<bool NewSymmetric>
    using setSymmetry =
        GpuSuperclusterNbListNeighborhoodConfig<NcMax, ISize, JSize, SuperclusterSize, Compress, NewSymmetric>;

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
    template<bool Symmetric>
    using setSymmetry     = GpuSuperclusterNbListNeighborhood<typename Config::template setSymmetry<Symmetric>>;
    using withSymmetry    = setSymmetry<true>;
    using withoutSymmetry = setSymmetry<false>;

    static constexpr unsigned ncMax            = Config::ncMax;
    static constexpr unsigned iSize            = Config::iSize;
    static constexpr unsigned jSize            = Config::jSize;
    static constexpr unsigned superclusterSize = Config::superclusterSize;
    static constexpr bool compress             = Config::compress;
    static constexpr bool symmetric            = Config::symmetric;

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
            thrust::device_vector<std::uint32_t>(previousSize),
            thrust::device_vector<SuperclusterInfo>(numISuperclusters)};

        if (numISuperclusters == 0) return nbList;

        thrust::device_vector<typename Config::SuperclusterSplitMask> superclusterSplitMasks(numISuperclusters);
        {
            constexpr unsigned numThreads = 256;
            const unsigned numBlocks      = iceil(numISuperclusters, numThreads);
            initSuperclusterInfo<<<numBlocks, numThreads>>>(firstISupercluster, lastISupercluster,
                                                            rawPtr(nbList.superclusterInfo));
            checkGpuErrors(cudaGetLastError());
        }
        {
            constexpr unsigned numThreads = 256;
            const unsigned numBlocks      = iceil(groups.numGroups, numThreads);
            computeSuperclusterSplitMasks<Config><<<numBlocks, numThreads>>>(
                firstISupercluster, lastISupercluster, firstValidBody, groups, rawPtr(superclusterSplitMasks));
            checkGpuErrors(cudaGetLastError());
        }

        thrust::device_vector<Vec3<Tc>> jClusterBboxCenters(numJClusters), jClusterBboxSizes(numJClusters);

        {
            constexpr unsigned numThreads = 256;
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
            previousSize = requiredSize * 1.1;
#ifndef NDEBUG
            checkGpuErrors(cudaMemcpy(&requiredSize, &rawPtr(globalBuildData)->neighborDataSize,
                                      sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            assert(requiredSize <= nbList.neighborData.size());
#endif
        }

        thrust::stable_sort(thrust::device, nbList.superclusterInfo.begin(), nbList.superclusterInfo.end());

        return nbList;
    }

private:
    mutable std::size_t previousSize = 0;
};

} // namespace cstone::ijloop
