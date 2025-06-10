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
 * @brief Data structures and functions used for building the supercluster neighborhood
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include "cstone/compressneighbors.cuh"
#include "cstone/cuda/memory.cuh"
#include "cstone/reducearray.cuh"
#include "cstone/traversal/groups.hpp"
#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/traversal/ijloop/gpu_superclusternblist/common.cuh"
#include "cstone/tree/octree.hpp"

namespace cstone::ijloop::gpu_supercluster_nb_list_neighborhood_detail
{

enum struct BuildStatus
{
    success = 0,
    neighbor_list_overflow,
    neighbor_data_overflow,
};

struct GlobalBuildData
{
    //! @brief total size of neighbor data, atomically increased during build to "allocate" required storage for each
    //! supercluster during build in a pre-allocated array
    unsigned long long neighborDataSize;
    //! @brief global group index counter, atomically increased during build
    unsigned index;
    BuildStatus status;
};

/*! initialize supercluster indices, required as superclusterInfo is later reordered by descending number of neighbors
 * to schedule more expensive warps earlier for better load balancing
 *
 * @param[in]  firstISupercluster index of first supercluster, i.e., the one containing firstBody
 * @param[in]  lastISupercluster  index of last supercluster
 * @param[out] superclusterInfo   indices to be filled with values from [firstISupercluster, lastISupercluster)
 */
__global__ static void initSuperclusterInfo(const LocalIndex firstISupercluster,
                                            const LocalIndex lastISupercluster,
                                            SuperclusterInfo* superclusterInfo)
{
    const LocalIndex index = blockIdx.x * blockDim.x + threadIdx.x;

    const LocalIndex numISuperclusters = lastISupercluster - firstISupercluster;
    if (index < numISuperclusters) superclusterInfo[index].index = index + firstISupercluster;
}

/*! compute supercluster split masks based on the given groups; split superclusters will execute multiple tree
 * traversals, i.e., one for each subgroup
 *
 * @param[in]  firstISupercluster     index of first supercluster, i.e., the one containing firstBody
 * @param[in]  firstValidBody         index of first valid particle, particles before are ignored
 * @param[in]  groups                 particle group information
 * @param[out] superclusterSplitMasks binary masks per supercluster, with ones where superclusters are spanning group
 *                                    boundaries, zeros elsewhere (i.e., one bit per particle)
 */
template<class Config, class SplitMask>
__global__ void computeSuperclusterSplitMasksKernel(const LocalIndex firstValidBody,
                                                    const GroupView __grid_constant__ groups,
                                                    const LocalIndex firstISupercluster,
                                                    SplitMask* __restrict__ superclusterSplitMasks)
{
    const LocalIndex index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= groups.numGroups) return;

    const LocalIndex groupEnd      = groups.groupEnd[index] + firstValidBody;
    const LocalIndex splitPosition = groupEnd % Config::superclusterSize;

    // no action required if the group's end is aligned to a supercluster boundary
    if (splitPosition == 0) return;

    const LocalIndex supercluster = superclusterIndex<Config>(groupEnd);
    auto* splitMaskPtr            = &superclusterSplitMasks[supercluster - firstISupercluster];
    const auto splitMask          = SplitMask(1) << splitPosition;

    // atomic update as multiple groups can split the same supercluster
    atomicOr(splitMaskPtr, splitMask);
}

template<class Config>
util::UniqueDevicePtr<typename Config::SuperclusterParticleMask[]>
computeSuperclusterSplitMasks(const LocalIndex firstValidBody,
                              const GroupView& groups,
                              const LocalIndex firstISupercluster,
                              const LocalIndex numISuperclusters)
{
    auto superclusterSplitMasks   = util::deviceAlloc<typename Config::SuperclusterParticleMask[]>(numISuperclusters);
    constexpr unsigned numThreads = 256;
    const unsigned numBlocks      = iceil(groups.numGroups, numThreads);
    computeSuperclusterSplitMasksKernel<Config>
        <<<numBlocks, numThreads>>>(firstValidBody, groups, firstISupercluster, superclusterSplitMasks.get());
    checkGpuErrors(cudaGetLastError());

    return superclusterSplitMasks;
}

/*! compute bounding boxes and max. particle radii of j-clusters, i.e., neighbor clusters
 *
 * @param[in]  firstValidBody index of first valid particle, particles before are ignored
 * @param[in]  totalBodies    total number of particles, including invalid
 * @param[in]  x              particle x coordinates
 * @param[in]  y              particle y coordinates
 * @param[in]  z              particle z coordinates
 * @param[in]  h              particle smoothing lengths
 * @param[out] bboxCenters    j-cluster bounding box centers
 * @param[out] bboxSizes      j-cluster bounding box sizes
 * @param[out] rMax           max. particle radius (2 * h) in each j-cluster, computed iff Config::symmetric
 */
template<class Config, class Tc, class Th>
__global__ void computeJClusterBboxes(const LocalIndex firstValidBody,
                                      const LocalIndex totalBodies,
                                      const Tc* const __restrict__ x,
                                      const Tc* const __restrict__ y,
                                      const Tc* const __restrict__ z,
                                      const Th* const __restrict__ h,
                                      Vec3<Tc>* const __restrict__ bboxCenters,
                                      Vec3<Tc>* const __restrict__ bboxSizes,
                                      Th* const __restrict__ rMaxs)
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

    if constexpr (Config::symmetric)
    {
        const Th hi = h[std::max(std::min(i, totalBodies - 1), firstValidBody)];
        Th rMax     = 2 * hi;

#pragma unroll
        for (unsigned offset = Config::jSize / 2; offset >= 1; offset /= 2)
            rMax = std::max(shflDownSync(rMax, offset), rMax);

        if (i % Config::jSize == 0 && jCluster < numJClusters) rMaxs[jCluster] = rMax;
    }
}

/*! sort candidate neighbor indices, does not require a fixed number of items per thread in contrast to CUB warp sort
 *
 * @param[inout] sharedAllocator shared memory allocator for temporary storage
 * @param[inout] candidates      neighbor cluster indices to be sorted
 * @param[in]    numCandidates   number of neighbor cluster candidates
 */
template<unsigned NumSuperclustersPerBlock>
__device__ __forceinline__ void
sortCandidates(util::SharedMemAllocator& sharedAllocator, std::uint32_t* candidates, unsigned numCandidates)
{
    const unsigned laneIdx = laneIndex();

    auto histograms = sharedAllocator.alloc<unsigned[]>(128);
    auto tmp        = sharedAllocator.alloc<std::uint32_t[]>(numCandidates);

    for (unsigned i = laneIdx; i < 128; i += GpuConfig::warpSize)
        histograms[i] = 0;

    syncWarp();

    for (unsigned i = laneIdx; i < numCandidates; i += GpuConfig::warpSize)
    {
        const auto value = candidates[i];
        for (unsigned digit = 0; digit < 8; ++digit)
            atomicAdd(&histograms[digit * 16 + ((value >> (4 * digit)) & 0xf)], 1);
    }

    syncWarp();

    for (unsigned i = laneIdx; i < 128; i += GpuConfig::warpSize)
    {
        const unsigned hist = histograms[i];
        unsigned index      = inclusiveScanInt(hist) - hist;
        index -= shflSync(index, (i / 16) * 16);
        histograms[i] = index;
    }

    syncWarp();

    for (unsigned digit = 0; digit < 8; ++digit)
    {
        const std::uint32_t* in = digit % 2 ? tmp.get() : candidates;
        std::uint32_t* out      = digit % 2 ? candidates : tmp.get();

        for (unsigned i = laneIdx; i < numCandidates; i += GpuConfig::warpSize)
        {
            const auto value     = in[i];
            const unsigned index = atomicAdd(&histograms[digit * 16 + ((value >> (digit * 4)) & 0xf)], 1);
            out[index]           = value;
        }

        syncWarp();
    }
}

/*! filter neighbor cluster candidates based on particle-particle distance checks and remove double entries
 *
 * @param[inout] sharedAllocator shared memory allocator for temporary storage
 * @param[in]    box             domain box
 * @param[in]    firstValidBody  index of first valid particle, particle before are ignored
 * @param[in]    totalBodies     total number of particles, including invalid
 * @param[in]    x               particle x coordinates
 * @param[in]    y               particle y coordinates
 * @param[in]    z               particle z coordinates
 * @param[in]    h               particle smoothing lengths
 * @param[in]    searchExtFactor factor to extend search radius
 * @param[in]    iSupercluster   current supercluster index
 * @param[inout] jClusters       array of candidate indices to be pruned, pruning happens in-place
 * @param[inout] numCandidates   number of neighbor clusters
 */
template<class Config, unsigned NumSuperclustersPerBlock, bool UsePbc, class Tc, class Th>
__device__ __forceinline__ void pruneCandidates(util::SharedMemAllocator& sharedAllocator,
                                                const Box<Tc>& box,
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

    auto xis = sharedAllocator.alloc<Tc[]>(Config::superclusterSize);
    auto yis = sharedAllocator.alloc<Tc[]>(Config::superclusterSize);
    auto zis = sharedAllocator.alloc<Tc[]>(Config::superclusterSize);
    auto his = sharedAllocator.alloc<Th[]>(Config::superclusterSize);

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

/*! decide if a neighbor index should be included in the symmetric neighbor list
 *
 * @param[in] i     own index
 * @param[in] j     neighbor index
 * @param[in] first start index of traversed entities
 * @param[in] last  end index of traversed entities
 *
 * @return true if the neighbor j should be included in the neighbor list of i, else false
 */
constexpr __forceinline__ bool includeNbSymmetric(unsigned i, unsigned j, unsigned first, unsigned last)
{
    // larger blockSize leads to more consecutive neighbors in list and thus improved neighbor list compression ratio
    // and cache locality
    constexpr unsigned blockSize = 32;
    const bool s                 = (i / blockSize) % 2 == (j / blockSize) % 2;
    return (j < first) | (j >= last) | (i == j) | (i < j ? s : !s);
}

/*! collect neighbor cluster candidates by traversing the octree and comparing bounding boxes of clusters
 *
 * @param[inout] sharedAllocator     shared memory allocator for temporary storage
 * @param[in]    tree                octree
 * @param[in]    box                 domain box
 * @param[in]    firstValidBody      index of first valid particle, particles before are ignored
 * @param[in]    totalBodies         total number of particles
 * @param[in]    firstBody           index of first particle
 * @param[in]    lastBody            index of last particle
 * @param[in]    firstGroupParticle  index of first particle in current consecutive group
 * @param[in]    lastGroupParticle   index of last particle in current consecutive group
 * @param[in]    x                   particle x coordinates
 * @param[in]    y                   particle y coordinates
 * @param[in]    z                   particle z coordinates
 * @param[in]    h                   particle smoothing lengths
 * @param[in]    jClusterBboxCenters bounding box centers of j-clusters
 * @param[in]    jClusterBboxSizes   bounding box sizes of j-clusters
 * @param[in]    jClusterRMax        max. particle radii of j-clusters
 * @param[in]    nodeRMax            max. particle radii of tree nodes
 * @param[in]    globalPool          global memory pool
 * @param[inout] candidates          array of candidate neighbor cluster indices
 * @param[inout] numCandidates       number of candidate cluster indices
 * @param[in]    ncmax               max. number of neighbor clusters (upper bound for numCandidates)
 */
template<class Config, unsigned NumSuperclustersPerBlock, bool UsePbc, class Tc, class Th, class KeyType>
__device__ __forceinline__ void collectJClusterCandidates(util::SharedMemAllocator& sharedAllocator,
                                                          const OctreeNsView<Tc, KeyType>& tree,
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
                                                          const Vec3<Tc>* const __restrict__ jClusterBboxCenters,
                                                          const Vec3<Tc>* const __restrict__ jClusterBboxSizes,
                                                          const Th* const __restrict__ jClusterRMax,
                                                          const Th* const __restrict__ nodeRMax,
                                                          int* __restrict__ globalPool,
                                                          unsigned* candidates,
                                                          unsigned& numCandidates,
                                                          const unsigned ncmax)
{
    const unsigned laneIdx = laneIndex();
    assert(blockDim.x * blockDim.y == GpuConfig::warpSize);
    assert(blockDim.z == NumSuperclustersPerBlock);

    Vec3<Tc> bbMin = {std::numeric_limits<Tc>::max(), std::numeric_limits<Tc>::max(), std::numeric_limits<Tc>::max()};
    Vec3<Tc> bbMax = {std::numeric_limits<Tc>::lowest(), std::numeric_limits<Tc>::lowest(),
                      std::numeric_limits<Tc>::lowest()};
    Th groupRMax   = 0;
    assert(lastGroupParticle - firstGroupParticle > 0);

    for (unsigned i = firstGroupParticle + laneIdx; i < lastGroupParticle; i += GpuConfig::warpSize)
    {
        const Vec3<Tc> iPos = {x[i], y[i], z[i]};
        const Tc hBound     = Config::symmetric ? Tc(0) : h[i];
#pragma unroll
        for (unsigned d = 0; d < 3; ++d)
        {
            bbMin[d] = std::min(bbMin[d], iPos[d] - 2 * hBound);
            bbMax[d] = std::max(bbMax[d], iPos[d] + 2 * hBound);
        }
        if constexpr (Config::symmetric) groupRMax = std::max(groupRMax, 2 * h[i]);
    }
#pragma unroll
    for (unsigned d = 0; d < 3; ++d)
    {
        bbMin[d] = warpMin(bbMin[d]);
        bbMax[d] = warpMax(bbMax[d]);
    }
    if constexpr (Config::symmetric) groupRMax = warpMax(groupRMax) * tree.searchExtFactor;

    const Vec3<Tc> groupCenter = (bbMax + bbMin) * Tc(0.5);
    const Vec3<Tc> groupSize   = (bbMax - bbMin) * Tc(0.5) * (Config::symmetric ? 1.0f : tree.searchExtFactor);

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
                Vec3<Tc> jClusterSize         = jClusterBboxSizes[jCluster];
                if constexpr (Config::symmetric)
                {
                    const Tc rMaxBound = std::max(groupRMax, jClusterRMax[jCluster] * tree.searchExtFactor);
                    for (unsigned d = 0; d < 3; ++d)
                        jClusterSize[d] += rMaxBound;
                }
                isNeighbor &= cellOverlap<UsePbc>(jClusterCenter, jClusterSize, groupCenter, groupSize, box);
            }
        }

        const unsigned nbIndex    = exclusiveScanBool(isNeighbor);
        unsigned newNumCandidates = shflSync(numCandidates + nbIndex + isNeighbor, GpuConfig::warpSize - 1);
        if (newNumCandidates >= ncmax)
        {
            sortCandidates<NumSuperclustersPerBlock>(sharedAllocator, candidates, numCandidates);
            pruneCandidates<Config, NumSuperclustersPerBlock, UsePbc>(sharedAllocator, box, firstValidBody, totalBodies,
                                                                      x, y, z, h, (Th)tree.searchExtFactor,
                                                                      iSupercluster, candidates, numCandidates);
            newNumCandidates = shflSync(numCandidates + nbIndex + isNeighbor, GpuConfig::warpSize - 1);
        }
        if (isNeighbor & (numCandidates + nbIndex < ncmax)) candidates[numCandidates + nbIndex] = jCluster;

        numCandidates = newNumCandidates;
    };

    auto sharedPool = sharedAllocator.alloc<int[]>(GpuConfig::warpSize);

    int jClusterQueue; // warp queue for source jCluster indices
    volatile int* tempQueue = sharedPool.get();
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

    const auto overlaps = [&](const Vec3<Tc>& srcCenter, Vec3<Tc> srcSize, Th srcRMax)
    {
        if constexpr (Config::symmetric)
        {
            Tc rMaxBound = std::max(groupRMax, srcRMax * tree.searchExtFactor);
            for (unsigned d = 0; d < 3; ++d)
                srcSize[d] += rMaxBound;
        }
        return cellOverlap<UsePbc>(srcCenter, srcSize, groupCenter, groupSize, box);
    };

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

        const Vec3<Tc> curSrcCenter = centers[sourceQueue]; // Current source cell center
        const Vec3<Tc> curSrcSize   = sizes[sourceQueue];   // Current source cell center
        const Th curSrcRMax         = Config::symmetric ? nodeRMax[sourceQueue] : Th(0);
        const int childBegin        = childOffsets[sourceQueue]; // First child cell
        const bool isNode           = childBegin;
        const bool isClose          = overlaps(curSrcCenter, curSrcSize, curSrcRMax);
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

/*! filter neighbor cluster candidates based on particle-particle distance checks, remove double entries, and compute
 * cluster-cluster interaction bitmasks
 *
 * @param[inout] sharedAllocator shared memory allocator for temporary storage
 * @param[in]    box             domain box
 * @param[in]    firstValidBody  index of first valid particle, particle before are ignored
 * @param[in]    totalBodies     total number of particles, including invalid
 * @param[in]    x               particle x coordinates
 * @param[in]    y               particle y coordinates
 * @param[in]    z               particle z coordinates
 * @param[in]    h               particle smoothing lengths
 * @param[in]    searchExtFactor factor to extend search radius
 * @param[in]    iSupercluster   current supercluster index
 * @param[inout] jClusters       array of candidate indices to be pruned, pruning happens in-place
 * @param[out]   masks           array of cluster-cluster interaction bitmasks
 * @param[in]    numCandidates   number of neighbor cluster candidates
 * @param[out]   numJClusters    number of neighbor clusters
 */
template<class Config, unsigned NumSuperclustersPerBlock, bool UsePbc, class Tc, class Th>
__device__ __forceinline__ void pruneCandidatesAndComputeMasks(util::SharedMemAllocator& sharedAllocator,
                                                               const Box<Tc>& box,
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

    auto xis = sharedAllocator.alloc<Tc[]>(Config::superclusterSize);
    auto yis = sharedAllocator.alloc<Tc[]>(Config::superclusterSize);
    auto zis = sharedAllocator.alloc<Tc[]>(Config::superclusterSize);
    auto his = sharedAllocator.alloc<Th[]>(Config::superclusterSize);

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

/*! store neighbor index data in global memory
 *
 * @param[inout] sharedAllocator  shared memory allocator
 * @param[in]    jClusters        sorted array of neighbor cluster indices
 * @param[in]    masks            array of cluster-cluster interaction bitmasks
 * @param[out]   neighborData     global memory neighbor data array where (possibly compressed) neighbor indices will be
 * stored
 * @param[in]    neighborDataSize size of neighborData array to avoid out of bounds accesses
 * @param[inout] info             supercluster info, will be updated with proper data index
 * @param[inout] globalBuildData  global build data used to get a global memory region
 */
template<class Config, unsigned NumSuperclustersPerBlock>
__device__ __forceinline__ void storeNeighborData(util::SharedMemAllocator& sharedAllocator,
                                                  const std::uint32_t* const __restrict__ jClusters,
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

    auto compressedJClusters = sharedAllocator.alloc<std::uint32_t[]>(Config::compress ? info.neighborsCount : 0);

    if constexpr (Config::compress)
    {
        warpCompressNeighbors(jClusters, (char*)compressedJClusters.get(), info.neighborsCount);
        nbSize = compressedNeighborsSize((const char*)compressedJClusters.get());
    }

    const unsigned long long totalSize = nbSize + mSize;
    if (laneIdx == 0) info.dataIndex = atomicAdd(&globalBuildData->neighborDataSize, totalSize);
    info.dataIndex = shflSync(info.dataIndex, 0);

    for (unsigned n = laneIdx; n < mSize; n += GpuConfig::warpSize)
    {
        const auto index = info.dataIndex + n;
        if (index >= neighborDataSize)
        {
            globalBuildData->status = BuildStatus::neighbor_data_overflow;
            return;
        }
        neighborData[index] = masks[n];
    }

    for (unsigned n = laneIdx; n < nbSize; n += GpuConfig::warpSize)
    {
        const auto index = info.dataIndex + mSize + n;
        if (index >= neighborDataSize)
        {
            globalBuildData->status = BuildStatus::neighbor_data_overflow;
            return;
        }
        neighborData[index] = (Config::compress ? compressedJClusters.get() : jClusters)[n];
    }
}

/*! compute required shared memory amount
 *
 * @param[in] ncmax maximum number of neighbor clusters
 */
template<class Config, class Tc, class Th>
constexpr unsigned buildNbListSharedMemPerSupercluster(const unsigned ncmax)
{
    // storage requirements for uncompressed neighbor indices
    const unsigned jClustersSize = ncmax * sizeof(unsigned);
    // storage requirements for cluster-cluster interaction bitmasks
    const unsigned masksDataSize = masksSize<Config>(ncmax) * sizeof(std::uint32_t);

    // storage requirements for sortCandidates
    constexpr unsigned histogramsSize = 128 * sizeof(unsigned);
    const unsigned tmpSize            = ncmax * sizeof(std::uint32_t);

    // storage requirements for cached particle coordinates and radii
    constexpr unsigned xisSize = Config::superclusterSize * sizeof(Tc);
    constexpr unsigned yisSize = Config::superclusterSize * sizeof(Tc);
    constexpr unsigned zisSize = Config::superclusterSize * sizeof(Tc);
    constexpr unsigned hisSize = Config::superclusterSize * sizeof(Th);

    // storage requirements for tree traversal
    constexpr unsigned sharedPoolSize = GpuConfig::warpSize * sizeof(int);

    // storage requirements for temporary array used for compression
    const unsigned compressedJClustersSize = (Config::compress ? ncmax : 0) * sizeof(std::uint32_t);

    return jClustersSize + masksDataSize +
           std::max({compressedJClustersSize, histogramsSize + tmpSize + sharedPoolSize,
                     xisSize + yisSize + zisSize + hisSize + sharedPoolSize});
}

/*! main GPU kernel for building the supercluster neighbor list
 *
 * @param[in]    tree                   octree
 * @param[in]    box                    domain box
 * @param[in]    firstValidBody         index of first valid particle, particles before are ignored
 * @param[in]    totalBodies            total number of particles
 * @param[in]    firstBody              index of first particle
 * @param[in]    lastBody               index of last particle
 * @param[in]    x                      particle x coordinates
 * @param[in]    y                      particle y coordinates
 * @param[in]    z                      particle z coordinates
 * @param[in]    h                      particle smoothing lengths
 * @param[in]    jClusterBboxCenters    bounding box centers of j-clusters
 * @param[in]    jClusterBboxSizes      bounding box sizes of j-clusters
 * @param[in]    jClusterRMax           max. particle radii of j-clusters
 * @param[in]    nodeRMax               max. particle radii of tree nodes
 * @param[in]    ncmax                  max. number of neighbor clusters (upper bound for numCandidates)
 * @param[in]    superclusterSplitMasks binary masks per supercluster, with ones where superclusters are spanning group
 *                                      boundaries, zeros elsewhere (i.e., one bit per particle)
 * @param[out]   neighborData           global memory neighbor data array where (possibly compressed) neighbor indices
 * will be stored
 * @param[in]    neighborDataSize       size of neighborData array to avoid out of bounds accesses
 * @param[inout] superclusterInfo       supercluster info
 * @param[in]    numSuperClusters       number of superclusters
 * @param[in]    globalPool             global memory pool used during tree traversal
 * @param[inout] globalBuildData        global build data used to 'allocate' global memory regions per supercluster in a
 * pre-allocated array
 */
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
    const Vec3<Tc>* const __restrict__ jClusterBboxCenters,
    const Vec3<Tc>* const __restrict__ jClusterBboxSizes,
    const Th* const __restrict__ jClusterRMax,
    const Th* const __restrict__ nodeRMax,
    const unsigned ncmax,
    const typename Config::SuperclusterParticleMask* const __restrict__ superclusterSplitMasks,
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

    util::SharedMemAllocator sharedAllocator(buildNbListSharedMemPerSupercluster<Config, Tc, Th>(ncmax), threadIdx.z);

    auto jClusters = sharedAllocator.alloc<std::uint32_t[]>(ncmax);

    while (true)
    {
        unsigned index;
        if (laneIdx == 0) index = atomicAdd(&globalBuildData->index, 1);
        index = shflSync(index, 0);
        if (index >= numSuperClusters) return;

        SuperclusterInfo info = {.index = superclusterInfo[index].index, .neighborsCount = 0, .dataIndex = 0};

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
                sharedAllocator, tree, box, firstValidBody, totalBodies, firstBody, lastBody, firstGroupParticle,
                lastGroupParticle, x, y, z, h, jClusterBboxCenters, jClusterBboxSizes, jClusterRMax, nodeRMax,
                globalPool, jClusters.get(), numCandidates, ncmax);

            if (numCandidates > ncmax)
            {
                globalBuildData->status = BuildStatus::neighbor_list_overflow;
                return;
            }
        }

        auto masks = sharedAllocator.alloc<std::uint32_t[]>(masksSize<Config>(numCandidates));

        sortCandidates<NumSuperclustersPerBlock>(sharedAllocator, jClusters.get(), numCandidates);
        pruneCandidatesAndComputeMasks<Config, NumSuperclustersPerBlock, UsePbc>(
            sharedAllocator, box, firstValidBody, totalBodies, x, y, z, h, (Th)tree.searchExtFactor, info.index,
            jClusters.get(), masks.get(), numCandidates, info.neighborsCount);

        storeNeighborData<Config, NumSuperclustersPerBlock>(sharedAllocator, jClusters.get(), masks.get(), neighborData,
                                                            neighborDataSize, info, globalBuildData);

        if (laneIdx == 0) superclusterInfo[index] = info;
    }
}
} // namespace cstone::ijloop::gpu_supercluster_nb_list_neighborhood_detail
