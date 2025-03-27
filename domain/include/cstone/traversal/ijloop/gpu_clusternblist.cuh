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
#include <cassert>
#include <tuple>
#include <type_traits>

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include "cstone/compressneighbors.cuh"
#include "cstone/cuda/cub.hpp"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/reducearray.cuh"
#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/traversal/ijloop/atomic_update_ptr.cuh"
#include "cstone/traversal/ijloop/ijloop.hpp"
#include "cstone/tree/octree.hpp"

namespace cstone::ijloop
{

namespace gpu_cluster_nb_list_neighborhood_detail
{

constexpr __forceinline__ bool includeNbSymmetric(unsigned i, unsigned j, unsigned first, unsigned last)
{
    constexpr unsigned blockSize = 8;
    const bool s                 = (i / blockSize) % 2 == (j / blockSize) % 2;
    return (j < first) | (j >= last) | (i < j ? s : !s);
}

template<class Config>
using nbStoragePerICluster =
    std::integral_constant<std::size_t,
                           (Config::compress ? Config::ncMax / Config::expectedCompressionRate : Config::ncMax)>;

template<class Config, class Tc>
__global__ void gpuClusterNbListComputeBboxes(LocalIndex totalBodies,
                                              const Tc* const __restrict__ x,
                                              const Tc* const __restrict__ y,
                                              const Tc* const __restrict__ z,
                                              util::tuple<Vec3<Tc>, Vec3<Tc>>* const __restrict__ bboxes)
{
    static_assert(GpuConfig::warpSize % Config::jSize == 0);

    const unsigned laneIdx = laneIndex();
    const unsigned i       = threadIdx.x + blockDim.x * blockIdx.x;

    const Tc xi = x[std::min(i, totalBodies - 1)];
    const Tc yi = y[std::min(i, totalBodies - 1)];
    const Tc zi = z[std::min(i, totalBodies - 1)];

    const unsigned jClusters = iceil(totalBodies, Config::jSize);
    const unsigned bboxIdx   = i / Config::jSize;

    if constexpr (Config::jSize >= 3)
    {
        util::array<Tc, 3> bboxMin{xi, yi, zi};
        util::array<Tc, 3> bboxMax{xi, yi, zi};

        const Tc vMin = reduceArray<Config::jSize, false>(bboxMin, [](auto a, auto b) { return std::min(a, b); });
        const Tc vMax = reduceArray<Config::jSize, false>(bboxMax, [](auto a, auto b) { return std::max(a, b); });

        const Tc center = (vMax + vMin) * Tc(0.5);
        const Tc size   = (vMax - vMin) * Tc(0.5);

        const unsigned idx = laneIdx % Config::jSize;
        if (idx < 3 & bboxIdx < jClusters)
        {
            auto* box     = &bboxes[bboxIdx];
            Tc* centerPtr = (Tc*)box + idx;
            Tc* sizePtr   = centerPtr + 3;
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

        if (i % Config::jSize == 0 && bboxIdx < jClusters) bboxes[bboxIdx] = {center, size};
    }
}

template<class Config, unsigned NumWarpsPerBlock>
__device__ __forceinline__ void deduplicateAndStoreNeighbors(unsigned* iClusterNidx,
                                                             const unsigned iClusterNc,
                                                             unsigned* targetIClusterNidx,
                                                             unsigned* targetIClusterNc)
{
    assert(blockDim.z == NumWarpsPerBlock);
    const unsigned laneIdx = laneIndex();

    constexpr unsigned itemsPerWarp = (Config::ncMax + Config::ncMaxExtra) / GpuConfig::warpSize;
    unsigned items[itemsPerWarp];
#pragma unroll
    for (unsigned i = 0; i < itemsPerWarp; ++i)
    {
        const unsigned nb = laneIdx * itemsPerWarp + i;
        items[i]          = nb < iClusterNc ? iClusterNidx[nb] : unsigned(-1);
    }
    using WarpSort = cub::WarpMergeSort<unsigned, itemsPerWarp, GpuConfig::warpSize>;
    __shared__ typename WarpSort::TempStorage sortTmp[NumWarpsPerBlock];
    WarpSort(sortTmp[threadIdx.z]).Sort(items, std::less<unsigned>());

    unsigned prev = shflUpSync(items[itemsPerWarp - 1], 1);
    if (laneIdx == 0) prev = unsigned(-1);
    unsigned unique = 0;
#pragma unroll
    for (unsigned i = 0; i < itemsPerWarp; ++i)
    {
        const unsigned item = items[i];
        if (item != prev & item != unsigned(-1))
        {
            // the following loop implements basically items[unique] = item;
            // but enables scalar replacement of items[]
#pragma unroll
            for (unsigned j = 0; j < itemsPerWarp; ++j)
            {
                if (j == unique) items[unique] = item;
            }
            ++unique;
            prev = item;
        }
    }

    const unsigned totalUnique = inclusiveScanInt(unique);
    assert(totalUnique < Config::ncMax);
    const unsigned startIndex = totalUnique - unique;

    if constexpr (Config::compress)
    {
        // the following loop with if-condition is equivalent to
        // for (unsigned i = 0; i < unique; ++i)
        // but enables scalar replacement of items[]
#pragma unroll
        for (unsigned i = 0; i < itemsPerWarp; ++i)
        {
            if (i < unique)
            {
                const unsigned nb = startIndex + i;
                iClusterNidx[nb]  = items[i];
            }
        }
        const unsigned uniqueNeighbors = shflSync(totalUnique, GpuConfig::warpSize - 1);
        assert(uniqueNeighbors < Config::ncMax);
        warpCompressNeighbors(iClusterNidx, (char*)targetIClusterNidx, uniqueNeighbors);
    }
    else
    {
#pragma unroll
        for (unsigned i = 0; i < itemsPerWarp; ++i)
        {
            if (i < unique)
            {
                const unsigned nb      = startIndex + i;
                targetIClusterNidx[nb] = items[i];
            }
        }

        if (laneIdx == GpuConfig::warpSize - 1) *targetIClusterNc = startIndex + unique;
    }
}

template<class Config, unsigned NumWarpsPerBlock, bool UsePbc, class Tc, class Th, class KeyType>
__global__
    __launch_bounds__(GpuConfig::warpSize * NumWarpsPerBlock) void gpuClusterNbListBuild(const OctreeNsView<Tc, KeyType> __grid_constant__ tree,
                                               const Box<Tc> __grid_constant__ box,
                                               const LocalIndex totalBodies,
                                               const LocalIndex firstBody,
                                               const LocalIndex lastBody,
                                               const Tc* const __restrict__ x,
                                               const Tc* const __restrict__ y,
                                               const Tc* const __restrict__ z,
                                               const Th* const __restrict__ h,
                                               const util::tuple<Vec3<Tc>, Vec3<Tc>>* const __restrict__ jClusterBboxes,
                                               unsigned* const __restrict__ clusterNeighbors,
                                               unsigned* const __restrict__ clusterNeighborsCount,
                                               int* const __restrict__ globalPool,
                                               const Th maxH)
{
    static_assert(Config::ncMax % GpuConfig::warpSize == 0);
    static_assert((Config::ncMax + Config::ncMaxExtra) % GpuConfig::warpSize == 0);

    assert(blockDim.x == Config::iSize);
    assert(blockDim.y == GpuConfig::warpSize / Config::iSize);
    assert(blockDim.z == NumWarpsPerBlock);
    static_assert(NumWarpsPerBlock > 0 && Config::iSize * Config::jSize <= GpuConfig::warpSize);
    static_assert(GpuConfig::warpSize % (Config::iSize * Config::jSize) == 0);
    const unsigned laneIdx = laneIndex();

    volatile __shared__ int sharedPool[GpuConfig::warpSize * NumWarpsPerBlock];
    __shared__ unsigned nidx[NumWarpsPerBlock][GpuConfig::warpSize / Config::iSize][Config::ncMax + Config::ncMaxExtra];

    const unsigned numTargets    = iceil(lastBody - firstBody / Config::iSize * Config::iSize, GpuConfig::warpSize);
    const unsigned firstICluster = firstBody / Config::iSize;
    const unsigned lastICluster  = iceil(lastBody, Config::iSize);
    const unsigned numJClusters  = iceil(totalBodies, Config::jSize);

    const TreeNodeIndex* __restrict__ childOffsets   = tree.childOffsets;
    const TreeNodeIndex* __restrict__ internalToLeaf = tree.internalToLeaf;
    const LocalIndex* __restrict__ layout            = tree.layout;
    const Vec3<Tc>* __restrict__ centers             = tree.centers;
    const Vec3<Tc>* __restrict__ sizes               = tree.sizes;

    while (true)
    {
        unsigned target;
        if (laneIdx == 0) target = atomicAdd(&targetCounterGlob, 1);
        target = shflSync(target, 0);

        if (target >= numTargets) return;

        const unsigned iCluster = firstICluster + target * (GpuConfig::warpSize / Config::iSize) + threadIdx.y;

        const unsigned i = std::min(
            std::max(target * GpuConfig::warpSize + laneIdx + firstBody / Config::iSize * Config::iSize, firstBody),
            totalBodies - 1);
        const Vec3<Tc> iPos = {x[i], y[i], z[i]};
        const Th hi         = h[i];

        const Th hBound = Config::symmetric ? maxH : hi;
        Vec3<Tc> bbMin  = {iPos[0] - 2 * hBound, iPos[1] - 2 * hBound, iPos[2] - 2 * hBound};
        Vec3<Tc> bbMax  = {iPos[0] + 2 * hBound, iPos[1] + 2 * hBound, iPos[2] + 2 * hBound};
        Vec3<Tc> iClusterCenter, iClusterSize;
        if constexpr (Config::iSize == 1)
        {
            iClusterCenter = (bbMax + bbMin) * Tc(0.5);
            iClusterSize   = (bbMax - bbMin) * Tc(0.5);
        }
#pragma unroll
        for (unsigned n = 1, s = 0; n < GpuConfig::warpSize; n *= 2, ++s)
        {
            bbMin[0] = std::min(shflXorSync(bbMin[0], 1 << s), bbMin[0]);
            bbMin[1] = std::min(shflXorSync(bbMin[1], 1 << s), bbMin[1]);
            bbMin[2] = std::min(shflXorSync(bbMin[2], 1 << s), bbMin[2]);
            bbMax[0] = std::max(shflXorSync(bbMax[0], 1 << s), bbMax[0]);
            bbMax[1] = std::max(shflXorSync(bbMax[1], 1 << s), bbMax[1]);
            bbMax[2] = std::max(shflXorSync(bbMax[2], 1 << s), bbMax[2]);
            if (n == Config::iSize / 2)
            {
                iClusterCenter = (bbMax + bbMin) * Tc(0.5);
                iClusterSize   = (bbMax - bbMin) * Tc(0.5);
            }
        }

        const Vec3<Tc> targetCenter = (bbMax + bbMin) * Tc(0.5);
        const Vec3<Tc> targetSize   = (bbMax - bbMin) * Tc(0.5);

        const auto distSq = [&](const Vec3<Tc>& jPos)
        { return distanceSq<UsePbc>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box); };

        unsigned nc = 0;

        const auto checkNeighborhood = [&](const unsigned jCluster, const unsigned numLanesValid)
        {
            assert(numLanesValid > 0);

            const unsigned prevJCluster = shflUpSync(jCluster, 1);
            const bool validJCluster =
                laneIdx < numLanesValid & jCluster < numJClusters & (laneIdx == 0 | prevJCluster != jCluster);

            const auto [jClusterCenter, jClusterSize] =
                validJCluster
                    ? jClusterBboxes[jCluster]
                    : util::tuple<Vec3<Tc>, Vec3<Tc>>({std::numeric_limits<Tc>::max(), std::numeric_limits<Tc>::max(),
                                                       std::numeric_limits<Tc>::max()},
                                                      {Tc(0), Tc(0), Tc(0)});

            for (unsigned c = 0; c < GpuConfig::warpSize / Config::iSize; ++c)
            {
                const auto iClusterC = firstICluster + target * (GpuConfig::warpSize / Config::iSize) + c;
                if (iClusterC < firstICluster | iClusterC >= lastICluster) break;
                const auto iClusterCenterC = shflSync(iClusterCenter, c * Config::iSize);
                const auto iClusterSizeC   = shflSync(iClusterSize, c * Config::iSize);
                const unsigned ncC         = shflSync(nc, c * Config::iSize);
                const bool isNeighbor =
                    (validJCluster & iClusterC * Config::iSize / Config::jSize != jCluster &
                     jCluster * Config::jSize / Config::iSize != iClusterC &
                     (!Config::symmetric || includeNbSymmetric(iClusterC, jCluster * Config::jSize / Config::iSize,
                                                               firstICluster, lastICluster))) &&
                    norm2(minDistance(iClusterCenterC, iClusterSizeC, jClusterCenter, jClusterSize, box)) == 0;

                const unsigned nbIndex = exclusiveScanBool(isNeighbor);
                if (isNeighbor & ncC + nbIndex < Config::ncMax + Config::ncMaxExtra)
                    nidx[threadIdx.z][c][ncC + nbIndex] = jCluster;
                const unsigned newNbs = shflSync(nbIndex + isNeighbor, GpuConfig::warpSize - 1);
                if (threadIdx.y == c) nc += newNbs;
            }
        };

        int jClusterQueue; // warp queue for source jCluster indices
        volatile int* tempQueue = sharedPool + GpuConfig::warpSize * threadIdx.z;
        int* cellQueue          = globalPool + TravConfig::memPerWarp * (blockIdx.x * NumWarpsPerBlock + threadIdx.z);

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
            sourceQueue = spreadSeg8(sourceQueue);
            sourceIdx   = shflSync(sourceIdx, laneIdx / 8);

            const Vec3<Tc> curSrcCenter = centers[sourceQueue];      // Current source cell center
            const Vec3<Tc> curSrcSize   = sizes[sourceQueue];        // Current source cell center
            const int childBegin        = childOffsets[sourceQueue]; // First child cell
            const bool isNode           = childBegin;
            const bool isClose          = cellOverlap<UsePbc>(curSrcCenter, curSrcSize, targetCenter, targetSize, box);
            const bool isSource         = sourceIdx < numSources; // Source index is within bounds
            const bool isDirect         = isClose && !isNode && isSource;
            const int leafIdx           = isDirect ? internalToLeaf[sourceQueue] : 0; // the cstone leaf index

            // Split
            const bool isSplit     = isNode && isClose && isSource; // Source cell must be split
            const int numChildLane = exclusiveScanBool(isSplit);    // Exclusive scan of numChild
            const int numChildWarp = reduceBool(isSplit);           // Total numChild of current warp
            sourceOffset +=
                imin(GpuConfig::warpSize / 8, numSources - sourceOffset);       // advance current level stack pointer
            int childIdx = oldSources + numSources + newSources + numChildLane; // Child index of current lane
            if (isSplit) cellQueue[ringAddr(childIdx)] = childBegin;            // Queue child cells for next level
            newSources += numChildWarp; // Increment source cell count for next loop

            // check for cellQueue overflow
            const unsigned stackUsed = newSources + numSources - sourceOffset; // current cellQueue size
            if (stackUsed > TravConfig::memPerWarp) return;                    // Exit if cellQueue overflows

            // Direct
            const int firstJCluster = layout[leafIdx] / Config::jSize;
            const int numJClusters  = (iceil(layout[leafIdx + 1], Config::jSize) - firstJCluster) &
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
                    checkNeighborhood(jClusterIdx, GpuConfig::warpSize);
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
                        checkNeighborhood(jClusterQueue, GpuConfig::warpSize);
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
            checkNeighborhood(jClusterQueue, jClusterFillLevel);

        for (unsigned c = 0; c < GpuConfig::warpSize / Config::iSize; ++c)
        {
            unsigned ncc      = shflSync(nc, c * Config::iSize);
            const unsigned ic = shflSync(iCluster, c * Config::iSize);
            if (ic < firstICluster | ic >= lastICluster) continue;

            const unsigned i    = std::min(ic * Config::iSize + threadIdx.x, totalBodies - 1);
            const Vec3<Tc> iPos = {x[i], y[i], z[i]};
            const Th hi         = h[i];

            const auto posDiff = [&](const Vec3<Tc>& jPos)
            {
                Vec3<Tc> ijPosDiff = {iPos[0] - jPos[0], iPos[1] - jPos[1], iPos[2] - jPos[2]};
                if constexpr (UsePbc)
                {
                    ijPosDiff[0] -=
                        (box.boundaryX() == BoundaryType::periodic) * box.lx() * std::rint(ijPosDiff[0] * box.ilx());
                    ijPosDiff[1] -=
                        (box.boundaryY() == BoundaryType::periodic) * box.ly() * std::rint(ijPosDiff[1] * box.ily());
                    ijPosDiff[2] -=
                        (box.boundaryZ() == BoundaryType::periodic) * box.lz() * std::rint(ijPosDiff[2] * box.ilz());
                }
                return ijPosDiff;
            };

            constexpr unsigned threadsPerInteraction = Config::iSize * Config::jSize;
            constexpr unsigned jBlocksPerWarp        = GpuConfig::warpSize / threadsPerInteraction;
            const GpuConfig::ThreadMask threadMask   = threadsPerInteraction == GpuConfig::warpSize
                                                           ? ~GpuConfig::ThreadMask(0)
                                                           : (GpuConfig::ThreadMask(1) << threadsPerInteraction) - 1;
            const GpuConfig::ThreadMask jBlockMask   = threadMask
                                                     << (threadsPerInteraction * (laneIdx / threadsPerInteraction));
            unsigned prunedNcc = 0;
            for (unsigned n = 0; n < imin(ncc, Config::ncMax + Config::ncMaxExtra); n += jBlocksPerWarp)
            {
                const unsigned nb = n + threadIdx.y / Config::jSize;
                const unsigned jCluster =
                    nb < imin(ncc, Config::ncMax + Config::ncMaxExtra) ? nidx[threadIdx.z][c][nb] : ~0u;
                const unsigned j         = jCluster * Config::jSize + threadIdx.y % Config::jSize;
                const Vec3<Tc> jPos      = j < totalBodies
                                               ? Vec3<Tc>{x[j], y[j], z[j]}
                                               : Vec3<Tc>{std::numeric_limits<Tc>::max(), std::numeric_limits<Tc>::max(),
                                                          std::numeric_limits<Tc>::max()};
                const Th hj              = j < totalBodies ? h[j] : 0;
                const Vec3<Tc> ijPosDiff = posDiff(jPos);
                const Th d2              = norm2(ijPosDiff);
                const Th iRadiusSq       = Th(4) * hi * hi;
                const Th jRadiusSq       = Th(4) * hj * hj;
                const bool keep       = ballotSync(d2 < iRadiusSq | (Config::symmetric & d2 < jRadiusSq)) & jBlockMask;
                const unsigned offset = exclusiveScanBool(keep & (laneIdx % threadsPerInteraction == 0));

                if ((laneIdx % threadsPerInteraction == 0) & keep) nidx[threadIdx.z][c][prunedNcc + offset] = jCluster;

                prunedNcc += shflSync(offset + keep, (GpuConfig::warpSize - 1) / (Config::iSize * Config::jSize) *
                                                         (Config::iSize * Config::jSize));
            }

            ncc = prunedNcc;

            deduplicateAndStoreNeighbors<Config, NumWarpsPerBlock>(
                nidx[threadIdx.z][c], ncc,
                &clusterNeighbors[(ic - firstICluster) * nbStoragePerICluster<Config>::value],
                &clusterNeighborsCount[ic - firstICluster]);
        }
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
    assert(blockDim.x == Config::iSize);

    if constexpr (std::conjunction_v<std::is_same<T0, T>...> && sizeof...(T) < GpuConfig::warpSize / Config::iSize &&
                  std::is_same<Postamble, detail::EmptyPostamble>())
    {
        const T0 res =
            reduceTuple<GpuConfig::warpSize / Config::iSize, true>(tuple,
                                                                   [](auto result, auto const& value)
                                                                   {
                                                                       detail::updateResultImpl(result, value);
                                                                       return result;
                                                                   });
        if ((threadIdx.y <= sizeof...(T)) & store)
        {
            auto* ptr = dynamicTupleGet(ptrs, threadIdx.y);
            if constexpr (Config::symmetric)
                atomicUpdatePtr(&ptr[index], res);
            else
                ptr[index] = detail::unwrapModifiersImpl(res);
        }
    }
    else
    {
#pragma unroll
        for (unsigned offset = GpuConfig::warpSize / 2; offset >= Config::iSize; offset /= 2)
            util::for_each_tuple([&](auto& t) { detail::updateResultImpl(t, shflDownSync(t, offset)); }, tuple);

        if ((threadIdx.y == 0) & store)
        {
            if constexpr (Config::symmetric)
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
    assert(blockDim.x == Config::iSize);

    if constexpr (std::conjunction_v<std::is_same<T0, T>...> && sizeof...(T) < Config::iSize)
    {
        const T0 res = reduceTuple<Config::iSize, false>(tuple,
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
        for (unsigned offset = Config::iSize / 2; offset >= 1; offset /= 2)
            util::for_each_tuple([&](auto& t) { detail::updateResultImpl(t, shflDownSync(t, offset)); }, tuple);

        if ((threadIdx.x == 0) & store)
            util::for_each_tuple([index](auto* ptr, auto const& t) { atomicUpdatePtr(&ptr[index], t); }, ptrs, tuple);
    }
}

template<class Config,
         unsigned NumWarpsPerBlock,
         bool UsePbc,
         class Tc,
         class Th,
         class In,
         class Out,
         class Interaction,
         class Postamble>
__global__ __launch_bounds__(GpuConfig::warpSize* NumWarpsPerBlock) void gpuClusterNbListNeighborhoodKernel(
    const Box<Tc> __grid_constant__ box,
    const LocalIndex totalBodies,
    const LocalIndex firstBody,
    const LocalIndex lastBody,
    const Tc* __restrict__ x,
    const Tc* __restrict__ y,
    const Tc* __restrict__ z,
    const Th* __restrict__ h,
    const In __grid_constant__ input,
    const Out __grid_constant__ output,
    const Interaction interaction,
    const Postamble postamble,
    const LocalIndex* __restrict__ clusterNeighbors,
    const unsigned* __restrict__ clusterNeighborsCount)
{
    static_assert(Config::ncMax % GpuConfig::warpSize == 0);

    assert(blockDim.x == Config::iSize);
    assert(blockDim.y == GpuConfig::warpSize / Config::iSize);
    static_assert(NumWarpsPerBlock > 0 && Config::iSize * Config::jSize <= GpuConfig::warpSize);
    static_assert(GpuConfig::warpSize % (Config::iSize * Config::jSize) == 0);
    assert(blockDim.z == NumWarpsPerBlock);
    const unsigned laneIdx = laneIndex();

    const unsigned firstICluster = firstBody / Config::iSize;
    const unsigned lastICluster  = iceil(lastBody, Config::iSize);
    const unsigned iCluster      = blockIdx.x * NumWarpsPerBlock + threadIdx.z + firstICluster;

    if (iCluster >= lastICluster) return;

    const unsigned i = iCluster * Config::iSize + threadIdx.x;

    __shared__ unsigned nidxBuffer[NumWarpsPerBlock][Config::ncMax];
    unsigned* const nidx = nidxBuffer[threadIdx.z];

    unsigned iClusterNeighborsCount;

    if constexpr (Config::compress)
    {
        warpDecompressNeighbors(
            (const char*)&clusterNeighbors[(iCluster - firstICluster) * nbStoragePerICluster<Config>::value], nidx,
            iClusterNeighborsCount);
    }
    else
    {
        iClusterNeighborsCount = imin(clusterNeighborsCount[iCluster - firstICluster], Config::ncMax);
#pragma unroll
        for (unsigned nb = laneIdx; nb < iClusterNeighborsCount; nb += GpuConfig::warpSize)
            nidx[nb] = clusterNeighbors[(iCluster - firstICluster) * nbStoragePerICluster<Config>::value + nb];
    }

    const auto iData =
        i < totalBodies ? loadParticleData(x, y, z, h, input, i) : dummyParticleData(x, y, z, h, input, i);

    using result_t = std::decay_t<decltype(interaction(iData, iData, Vec3<Tc>(), Tc(0)))>;

    static_assert(!Config::symmetric ||
                      std::is_same<std::decay_t<decltype(postamble(iData, unwrapModifiers(result_t())))>,
                                   decltype(unwrapModifiers(result_t()))>(),
                  "postamble that changes the result type is not supported in combination with symmetric neighborhood");

    result_t result                      = {};
    const auto computeClusterInteraction = [&](const unsigned jCluster, const bool self)
    {
        const unsigned j = jCluster * Config::jSize + threadIdx.y % Config::jSize;
        result_t jResult = {};
        if (i < totalBodies & j < totalBodies & (!Config::symmetric | !self | (i <= j)))
        {
            const auto jData = loadParticleData(x, y, z, h, input, j);

            const auto [ijPosDiff, distSq] = posDiffAndDistSq(UsePbc, box, iData, jData);
            const Th iRadiusSq             = radiusSq(iData);
            const Th jRadiusSq             = radiusSq(jData);
            const auto ijInteraction       = interaction(iData, jData, ijPosDiff, distSq);
            if (distSq < iRadiusSq) updateResult(result, ijInteraction);
            if constexpr (Config::symmetric)
            {
                if (distSq < jRadiusSq & (!self | (i != j)))
                {
                    const auto jiInteraction =
                        selectSymmetric(ijInteraction, interaction(jData, iData, -ijPosDiff, distSq));
                    updateResult(jResult, jiInteraction);
                }
            }
        }
        if constexpr (Config::symmetric) storeTupleJSum<Config>(jResult, output, j, j >= firstBody & j < lastBody);
    };

    constexpr unsigned jClustersPerWarp     = GpuConfig::warpSize / Config::iSize / Config::jSize;
    constexpr unsigned overlappingJClusters = Config::iSize <= Config::jSize ? 1 : Config::iSize / Config::jSize;
#pragma unroll
    for (unsigned overlapping = 0; overlapping < overlappingJClusters; overlapping += jClustersPerWarp)
    {
        const unsigned o        = overlapping + threadIdx.y / Config::jSize;
        const unsigned jCluster = o < overlappingJClusters ? iCluster * Config::iSize / Config::jSize + o : ~0u;
        computeClusterInteraction(jCluster, true);
    }
    for (unsigned jc = 0; jc < iClusterNeighborsCount; jc += jClustersPerWarp)
    {
        const unsigned jcc      = jc + threadIdx.y / Config::jSize;
        const unsigned jCluster = jcc < iClusterNeighborsCount ? nidx[jcc] : ~0u;
        computeClusterInteraction(jCluster, false);
    }

    storeTupleISum<Config>(result, output, i, i >= firstBody & i < lastBody, postamble, iData);
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
    const LocalIndex i = blockIdx.x * blockDim.x + threadIdx.x + firstBody;
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
    const LocalIndex i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > totalBodies) return;

    const auto iData  = loadParticleData(x, y, z, h, input, i);
    const auto result = util::tupleMap([&](auto* ptr) { return ptr[i]; }, output);
    storeParticleData(output, i, postamble(iData, result));
}

template<class Config, class Tc, class Th>
struct GpuClusterNbListNeighborhoodImpl
{
    Box<Tc> box;
    LocalIndex totalBodies, firstBody, lastBody;
    const Tc *x, *y, *z;
    const Th* h;
    thrust::device_vector<LocalIndex> clusterNeighbors;
    thrust::device_vector<unsigned> clusterNeighborsCount;

    template<class... In, class... Out, class Interaction, class Postamble>
    void ijLoop(std::tuple<In*...> const& input,
                std::tuple<Out*...> const& output,
                Interaction&& interaction,
                Postamble&& postamble) const
    {
        const LocalIndex numBodies = lastBody - firstBody;
        if (Config::symmetric)
        {
            constexpr unsigned threads = 128;
            const unsigned numBlocks   = iceil(numBodies, threads);
            initResult<<<numBlocks, threads>>>(firstBody, lastBody, x, y, z, h, makeConstRestrict(input), output,
                                               std::forward<Interaction>(interaction));
            checkGpuErrors(cudaGetLastError());
        }

        const LocalIndex firstICluster = firstBody / Config::iSize;
        const LocalIndex lastICluster  = iceil(lastBody, Config::iSize);
        const LocalIndex numIClusters  = lastICluster - firstICluster;

        constexpr unsigned threads          = 128;
        constexpr unsigned numWarpsPerBlock = threads / GpuConfig::warpSize;
        const dim3 blockSize                = {Config::iSize, GpuConfig::warpSize / Config::iSize, numWarpsPerBlock};
        const unsigned numBlocks            = iceil(numIClusters, numWarpsPerBlock);
        gpuClusterNbListNeighborhoodKernel<Config, numWarpsPerBlock, true><<<numBlocks, blockSize>>>(
            box, totalBodies, firstBody, lastBody, x, y, z, h, makeConstRestrict(input), output,
            std::forward<Interaction>(interaction), std::forward<Postamble>(postamble), rawPtr(clusterNeighbors),
            rawPtr(clusterNeighborsCount));
        checkGpuErrors(cudaGetLastError());

        if constexpr (Config::symmetric && !std::is_same<std::decay_t<Postamble>, detail::EmptyPostamble>())
        {
            constexpr unsigned threads = 128;
            const unsigned numBlocks   = iceil(totalBodies, threads);
            applyPostamble<<<numBlocks, threads>>>(totalBodies, x, y, z, h, makeConstRestrict(input), output,
                                                   std::forward<Postamble>(postamble));
        }
    }

    Statistics stats() const
    {
        return {.numBodies = lastBody - firstBody,
                .numBytes =
                    clusterNeighbors.size() * sizeof(typename decltype(clusterNeighbors)::value_type) +
                    clusterNeighborsCount.size() * sizeof(typename decltype(clusterNeighborsCount)::value_type)};
    }
};

template<unsigned NcMax                   = 256,
         unsigned ISize                   = 4,
         unsigned JSize                   = 4,
         unsigned ExpectedCompressionRate = 0,
         bool Symmetric                   = true>
struct GpuClusterNbListNeighborhoodConfig
{
    static constexpr unsigned ncMax                   = NcMax;
    static constexpr unsigned iSize                   = ISize;
    static constexpr unsigned jSize                   = JSize;
    static constexpr unsigned expectedCompressionRate = ExpectedCompressionRate;
    static constexpr bool compress                    = ExpectedCompressionRate > 1;
    static constexpr bool symmetric                   = Symmetric;
    static constexpr unsigned ncMaxExtra              = NcMax;

    template<unsigned NewNcMax>
    using withNcMax = GpuClusterNbListNeighborhoodConfig<NewNcMax, ISize, JSize, ExpectedCompressionRate, Symmetric>;

    template<unsigned NewISize, unsigned newJSize>
    using withClusterSize =
        GpuClusterNbListNeighborhoodConfig<NcMax, NewISize, newJSize, ExpectedCompressionRate, Symmetric>;
    template<unsigned NewExpectedCompressionRate>
    using withCompression =
        GpuClusterNbListNeighborhoodConfig<NcMax, ISize, JSize, NewExpectedCompressionRate, Symmetric>;
    using withoutCompression = withCompression<0>;
    using withSymmetry       = GpuClusterNbListNeighborhoodConfig<NcMax, ISize, JSize, ExpectedCompressionRate, true>;
    using withoutSymmetry    = GpuClusterNbListNeighborhoodConfig<NcMax, ISize, JSize, ExpectedCompressionRate, false>;
};

} // namespace gpu_cluster_nb_list_neighborhood_detail

template<class Config = gpu_cluster_nb_list_neighborhood_detail::GpuClusterNbListNeighborhoodConfig<>>
struct GpuClusterNbListNeighborhood
{
    template<unsigned NcMax>
    using withNcMax = GpuClusterNbListNeighborhood<typename Config::withNcMax<NcMax>>;
    template<unsigned ISize, unsigned JSize>
    using withClusterSize = GpuClusterNbListNeighborhood<typename Config::withClusterSize<ISize, JSize>>;
    template<unsigned ExpectedCompressionRate>
    using withCompression    = GpuClusterNbListNeighborhood<typename Config::withCompression<ExpectedCompressionRate>>;
    using withoutCompression = GpuClusterNbListNeighborhood<typename Config::withoutCompression>;
    using withSymmetry       = GpuClusterNbListNeighborhood<typename Config::withSymmetry>;
    using withoutSymmetry    = GpuClusterNbListNeighborhood<typename Config::withoutSymmetry>;

    template<class Tc, class KeyType, class Th>
    gpu_cluster_nb_list_neighborhood_detail::GpuClusterNbListNeighborhoodImpl<Config, Tc, Th>
    build(const OctreeNsView<Tc, KeyType>& tree,
          const Box<Tc>& box,
          const LocalIndex totalBodies,
          const GroupView& groups,
          const Tc* x,
          const Tc* y,
          const Tc* z,
          const Th* h) const
    {
        using namespace gpu_cluster_nb_list_neighborhood_detail;

        const LocalIndex firstICluster = groups.firstBody / Config::iSize;
        const LocalIndex lastICluster  = iceil(groups.lastBody, Config::iSize);
        const LocalIndex numIClusters  = lastICluster - firstICluster;
        const LocalIndex numJClusters  = iceil(totalBodies, Config::jSize);

        thrust::device_vector<util::tuple<Vec3<Tc>, Vec3<Tc>>> jClusterBboxes(numJClusters);

        GpuClusterNbListNeighborhoodImpl<Config, Tc, Th> nbList{
            box,
            totalBodies,
            groups.firstBody,
            groups.lastBody,
            x,
            y,
            z,
            h,
            thrust::device_vector<LocalIndex>(nbStoragePerICluster<Config>::value * numIClusters),
            thrust::device_vector<unsigned>(Config::compress ? 0 : numIClusters)};

        {
            constexpr unsigned numThreads = 128;
            unsigned numBlocks            = iceil(totalBodies, numThreads);
            gpuClusterNbListComputeBboxes<Config>
                <<<numBlocks, numThreads>>>(totalBodies, x, y, z, rawPtr(jClusterBboxes));
            checkGpuErrors(cudaGetLastError());
        }
        {
            constexpr unsigned numThreads       = 64;
            constexpr unsigned numWarpsPerBlock = numThreads / GpuConfig::warpSize;
            unsigned numBlocks = GpuConfig::smCount * (TravConfig::numWarpsPerSm / (numThreads / GpuConfig::warpSize));
            const dim3 blockSize = {Config::iSize, GpuConfig::warpSize / Config::iSize, numWarpsPerBlock};
            thrust::device_vector<int> pool(TravConfig::memPerWarp * numWarpsPerBlock * numBlocks);
            Th maxH = 0;
            if constexpr (Config::symmetric)
                maxH = thrust::reduce(thrust::device, h, h + totalBodies, Th(0), thrust::maximum<Th>());

            resetTraversalCounters<<<1, 1>>>();
            gpuClusterNbListBuild<Config, numWarpsPerBlock, true><<<numBlocks, blockSize>>>(
                tree, box, totalBodies, groups.firstBody, groups.lastBody, x, y, z, h, rawPtr(jClusterBboxes),
                rawPtr(nbList.clusterNeighbors), rawPtr(nbList.clusterNeighborsCount), rawPtr(pool), maxH);
            checkGpuErrors(cudaGetLastError());

            checkGpuErrors(cudaDeviceSynchronize());
        }
        return nbList;
    }
};

} // namespace cstone::ijloop
