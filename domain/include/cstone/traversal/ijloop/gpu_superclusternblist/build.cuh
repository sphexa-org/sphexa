/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Data structures and functions used for building the supercluster neighborhood
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#include "cstone/cuda/memory.cuh"
#include "cstone/execution.hpp"
#include "cstone/reducearray.cuh"
#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/traversal/groups.hpp"
#include "cstone/traversal/ijloop/compressneighbors.cuh"
#include "cstone/traversal/ijloop/gpu_superclusternblist/common.cuh"
#include "cstone/traversal/ijloop/upsweep.cuh"
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
    //! @brief maximum number of cluster neighbors
    unsigned maxNeighbors;
};

template<class Tc>
struct JClusterBboxAsymmetric
{
    Vec3<Tc> center, size;
};

template<class Tc>
struct JClusterBboxSymmetric : JClusterBboxAsymmetric<Tc>
{
    Tc rMax;
};

template<class Config, class Tc>
using JClusterBbox = std::conditional_t<Config::symmetric, JClusterBboxSymmetric<Tc>, JClusterBboxAsymmetric<Tc>>;

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
template<class Config, class Tc, class ThP>
__global__ void computeJClusterBboxesKernel(const LocalIndex firstValidBody,
                                            const LocalIndex totalBodies,
                                            const Tc* const __restrict__ x,
                                            const Tc* const __restrict__ y,
                                            const Tc* const __restrict__ z,
                                            const ThP h,
                                            JClusterBbox<Config, Tc>* const __restrict__ bboxes)
{
    static_assert(GpuConfig::warpSize % Config::jSize == 0);

    const unsigned i = threadIdx.x + blockDim.x * blockIdx.x;

    const Tc xi = x[std::max(std::min(i, totalBodies - 1), firstValidBody)];
    const Tc yi = y[std::max(std::min(i, totalBodies - 1), firstValidBody)];
    const Tc zi = z[std::max(std::min(i, totalBodies - 1), firstValidBody)];

    const unsigned numJClusters = jClusterIndex<Config>(totalBodies - 1) + 1;
    const unsigned jCluster     = jClusterIndex<Config>(i);

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
        bboxes[jCluster].center = center;
        bboxes[jCluster].size   = size;
    }

    if constexpr (Config::symmetric)
    {
        using Th = std::remove_cvref_t<std::remove_pointer_t<ThP>>;
        Th hi;
        if constexpr (std::is_pointer_v<ThP>)
            hi = h[std::max(std::min(i, totalBodies - 1), firstValidBody)];
        else
            hi = h;
        Th rMax = 2 * hi;

#pragma unroll
        for (unsigned offset = Config::jSize / 2; offset >= 1; offset /= 2)
            rMax = std::max(shflDownSync(rMax, offset), rMax);

        if (i % Config::jSize == 0 && jCluster < numJClusters) bboxes[jCluster].rMax = rMax;
    }
}

template<class Config, class Tc, class ThP>
util::UniqueDevicePtr<JClusterBbox<Config, Tc>[]> computeJClusterBboxes(const execution::Gpu exec,
                                                                        const LocalIndex firstValidBody,
                                                                        const LocalIndex totalBodies,
                                                                        const Tc* const __restrict__ x,
                                                                        const Tc* const __restrict__ y,
                                                                        const Tc* const __restrict__ z,
                                                                        const ThP h)
{
    const LocalIndex numJClusters = jClusterIndex<Config>(totalBodies - 1) + 1;
    auto jClusterBboxes           = util::deviceAlloc<JClusterBbox<Config, Tc>[]>(exec, numJClusters);
    constexpr unsigned numThreads = 256;
    unsigned numBlocks            = iceil(numJClusters * Config::jSize, numThreads);
    computeJClusterBboxesKernel<Config>
        <<<numBlocks, numThreads, 0, exec>>>(firstValidBody, totalBodies, x, y, z, h, jClusterBboxes.get());
    checkGpuErrors(cudaGetLastError());
    return jClusterBboxes;
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

/*! store neighbor index data in global memory
 *
 * @param[in]    jClusters           sorted array of neighbor cluster indices
 * @param[in]    masks               array of cluster-cluster interaction bitmasks
 * @param[out]   neighborData        global memory neighbor data array where (possibly compressed) neighbor indices will
 *                                   be stored
 * @param[in]    maxNeighborDataSize max size of neighborData array to avoid out of bounds accesses
 * @param[inout] neighborDataSize    current size of neighborData array
 * @param[inout] info                supercluster info, will be updated with proper data index
 */
template<class Config, unsigned NumSuperclustersPerBlock>
__device__ __forceinline__ bool storeNeighborData(std::uint32_t* const __restrict__ jClusters,
                                                  const unsigned jClusterBytes,
                                                  const std::uint32_t* const __restrict__ masks,
                                                  const unsigned ncmax,
                                                  std::uint32_t* const __restrict__ neighborData,
                                                  const std::size_t maxNeighborDataSize,
                                                  unsigned long long* __restrict__ neighborDataSize,
                                                  SuperclusterInfo& info)
{
    const unsigned laneIdx = laneIndex();
    assert(blockDim.x * blockDim.y == GpuConfig::warpSize);
    assert(blockDim.z == NumSuperclustersPerBlock);

    const unsigned mSize  = masksSize<Config>(std::min(info.neighborsCount, ncmax));
    const unsigned nbSize = (jClusterBytes + sizeof(std::uint32_t) - 1) / sizeof(std::uint32_t);

    const unsigned long long totalSize = nbSize + mSize;
    if (laneIdx == 0) info.dataIndex = atomicAdd(neighborDataSize, totalSize);
    info.dataIndex = shflSync(info.dataIndex, 0);

    if (info.dataIndex + mSize + nbSize > maxNeighborDataSize) return false;

    for (unsigned n = laneIdx; n < mSize; n += GpuConfig::warpSize)
        neighborData[info.dataIndex + n] = masks[n];

    for (unsigned n = laneIdx; n < nbSize; n += GpuConfig::warpSize)
        neighborData[info.dataIndex + mSize + n] = jClusters[n];

    return true;
}

template<class Config, class Tc, class ThP>
__device__ __forceinline__ auto loadSuperclusterParticleData(const LocalIndex firstBody,
                                                             const LocalIndex lastBody,
                                                             const Tc* const __restrict__ x,
                                                             const Tc* const __restrict__ y,
                                                             const Tc* const __restrict__ z,
                                                             const ThP h,
                                                             const float searchExtFactor)
{
    constexpr unsigned warpsPerSupercluster = Config::superclusterSize / GpuConfig::warpSize;
    using Th                                = std::remove_cvref_t<std::remove_pointer_t<ThP>>;
    const unsigned laneIdx                  = laneIndex();

    std::array<Vec3<Tc>, warpsPerSupercluster> iPos;
    std::array<Th, warpsPerSupercluster> iRadius;

    for (unsigned w = 0; w < warpsPerSupercluster; ++w)
    {
        const unsigned i = std::min(firstBody + w * GpuConfig::warpSize + laneIdx, lastBody - 1);
        iPos[w]          = {x[i], y[i], z[i]};
        iRadius[w]       = 2 * loadAtIndexIfPtr(h, i) * searchExtFactor;
    }
    return std::make_tuple(iPos, iRadius);
}

template<class Config, std::size_t WarpsPerSupercluster, class Tc, class Th>
__device__ __forceinline__ std::tuple<Vec3<Tc>, Vec3<Tc>, Th>
superclusterBoundingBox(const std::array<Vec3<Tc>, WarpsPerSupercluster>& iPos,
                        const std::array<Th, WarpsPerSupercluster>& iRadius)
{
    Vec3<Tc> bBoxMin = {std::numeric_limits<Tc>::max(), std::numeric_limits<Tc>::max(), std::numeric_limits<Tc>::max()};
    Vec3<Tc> bBoxMax = {std::numeric_limits<Tc>::lowest(), std::numeric_limits<Tc>::lowest(),
                        std::numeric_limits<Tc>::lowest()};
    Th maxParticleRadius = 0;
    for (unsigned w = 0; w < WarpsPerSupercluster; ++w)
    {
        const Tc iBbRadius = Config::symmetric ? 0 : iRadius[w];
        for (unsigned d = 0; d < 3; ++d)
        {
            bBoxMin[d] = std::min(bBoxMin[d], iPos[w][d] - iBbRadius);
            bBoxMax[d] = std::max(bBoxMax[d], iPos[w][d] + iBbRadius);
        }
        maxParticleRadius = std::max(maxParticleRadius, iRadius[w]);
    }

    for (unsigned d = 0; d < 3; ++d)
    {
        bBoxMin[d] = warpMin(bBoxMin[d]);
        bBoxMax[d] = warpMax(bBoxMax[d]);
    }
    maxParticleRadius = warpMax(maxParticleRadius);

    const Vec3<Tc> bBoxCenter = (bBoxMax + bBoxMin) * Tc(0.5);
    const Vec3<Tc> bBoxSize   = (bBoxMax - bBoxMin) * Tc(0.5);

    return {bBoxCenter, bBoxSize, maxParticleRadius};
}

/*! traverse the octree to find neighbor clusters for a supercluster
 *
 * @param[in]    tree                   octree
 * @param[in]    box                    domain box
 * @param[in]    firstValidBody         index of first valid particle
 * @param[in]    totalBodies            total number of particles
 * @param[in]    x                      particle x coordinates
 * @param[in]    y                      particle y coordinates
 * @param[in]    z                      particle z coordinates
 * @param[in]    h                      particle smoothing lengths
 * @param[in]    jClusterBboxes         bounding boxes of j-clusters
 * @param[in]    nodeRMax               max. particle radii of tree nodes
 * @param[in]    ncmax                  max. number of neighbor clusters
 * @param[in]    firstISupercluster     supercluster index offset
 * @param[in]    lastISupercluster      index of the last supercluster
 * @param[inout] jClusters              shared memory array for neighbor cluster indices
 * @param[inout] masks                  shared memory array for interaction bitmasks
 * @param[inout] info                   supercluster info
 */
template<class Config, bool UsePbc, class Tc, class ThP, class KeyType>
__device__ __forceinline__ unsigned
collectNeighborJClusters(const OctreeNsView<Tc, KeyType>& tree,
                         const Box<Tc>& box,
                         const LocalIndex firstValidBody,
                         const LocalIndex totalBodies,
                         const Tc* const __restrict__ x,
                         const Tc* const __restrict__ y,
                         const Tc* const __restrict__ z,
                         const ThP h,
                         const JClusterBbox<Config, Tc>* const __restrict__ jClusterBboxes,
                         const ThP nodeRMax,
                         const unsigned ncmax,
                         const unsigned firstISupercluster,
                         const unsigned lastISupercluster,
                         std::uint32_t* const jClusters,
                         std::uint32_t* const masks,
                         SuperclusterInfo& info)
{
    constexpr unsigned warpsPerSupercluster = Config::superclusterSize / GpuConfig::warpSize;

    using Th               = std::remove_cvref_t<std::remove_pointer_t<ThP>>;
    const unsigned laneIdx = laneIndex();

    const unsigned firstBody     = std::max(info.index * Config::superclusterSize, firstValidBody);
    const unsigned lastBody      = std::min((info.index + 1) * Config::superclusterSize, totalBodies);
    const unsigned iSupercluster = superclusterIndex<Config>(firstBody);

    const auto [iPos, iRadius] =
        loadSuperclusterParticleData<Config>(firstBody, lastBody, x, y, z, h, tree.searchExtFactor);
    const auto [bBoxCenter, bBoxSize, maxParticleRadius] = superclusterBoundingBox<Config>(iPos, iRadius);

    const auto overlapsInternalNode = [&](const TreeNodeIndex idx)
    {
        const Vec3<Tc> srcCenter = tree.centers[idx];
        const Vec3<Tc> srcSize   = tree.sizes[idx];
        const Th srcRadius       = Config::symmetric ? loadAtIndexIfPtr(nodeRMax, idx) * tree.searchExtFactor : Th(0);

        bool overlaps = false;
        for (unsigned w = 0; w < warpsPerSupercluster; ++w)
        {
            const Th maxRadius = std::max(iRadius[w], srcRadius);
            const auto distance =
                UsePbc ? minDistance(iPos[w], srcCenter, srcSize, box) : minDistance(iPos[w], srcCenter, srcSize);
            overlaps |= norm2(distance) < maxRadius * maxRadius;
        }

        return bool(anySync(overlaps));
    };

    typename Config::Compression compression(jClusters);

    unsigned jClusterQueue, previousJCluster = ~0u;
    const auto overlapsLeafNode = [&](const TreeNodeIndex idx)
    {
        const TreeNodeIndex leafIdx    = tree.internalToLeaf[idx];
        const LocalIndex firstJCluster = jClusterIndex<Config>(tree.layout[leafIdx] + firstValidBody);
        const LocalIndex lastJCluster  = tree.layout[leafIdx + 1] == tree.layout[leafIdx]
                                             ? 0
                                             : jClusterIndex<Config>(tree.layout[leafIdx + 1] + firstValidBody - 1) + 1;

        for (LocalIndex baseJCluster = firstJCluster; baseJCluster < lastJCluster; baseJCluster += GpuConfig::warpSize)
        {
            const LocalIndex jCluster = std::min(baseJCluster + laneIdx, lastJCluster - 1);
            bool bBoxesOverlap        = jCluster != previousJCluster;

            if constexpr (Config::symmetric)
            {
                const LocalIndex jSupercluster = superclusterIndex<Config>(jCluster * Config::jSize);
                if (bBoxesOverlap)
                {
                    bBoxesOverlap &=
                        includeNbSymmetric(iSupercluster, jSupercluster, firstISupercluster, lastISupercluster);
                }
            }

            if (bBoxesOverlap)
            {
                auto jClusterBBox = jClusterBboxes[jCluster];
                if constexpr (Config::symmetric)
                {
                    const Tc rMaxBound = std::max(Tc(maxParticleRadius), jClusterBBox.rMax * tree.searchExtFactor);
                    for (unsigned d = 0; d < 3; ++d)
                        jClusterBBox.size[d] += rMaxBound;
                }
                bBoxesOverlap &= cellOverlap<UsePbc>(jClusterBBox.center, jClusterBBox.size, bBoxCenter, bBoxSize, box);
            }

            for (unsigned lane = 0; lane < GpuConfig::warpSize; ++lane)
            {
                const unsigned jCluster = baseJCluster + lane;
                if (jCluster >= lastJCluster) break;
                if (!shflSync(bBoxesOverlap, lane)) continue;

                unsigned warpMask = 0;
                for (LocalIndex jClusterParticle = 0; jClusterParticle < Config::jSize; ++jClusterParticle)
                {
                    const LocalIndex j =
                        std::clamp(jCluster * Config::jSize + jClusterParticle, firstValidBody, totalBodies - 1);
                    const Vec3<Tc> jPos = {x[j], y[j], z[j]};
                    const Th jRadius    = Config::symmetric ? 2 * loadAtIndexIfPtr(h, j) * tree.searchExtFactor : Th(0);
                    const unsigned warpIndex = jClusterParticle / (Config::jSize / Config::numWarpsPerInteraction);

                    for (unsigned w = 0; w < warpsPerSupercluster; ++w)
                    {
                        const unsigned c   = laneIdx / Config::iSize + w * (GpuConfig::warpSize / Config::iSize);
                        const Th maxRadius = std::max(jRadius, iRadius[w]);
                        const bool jClusterOverlaps =
                            distanceSq<UsePbc>(jPos[0], jPos[1], jPos[2], iPos[w][0], iPos[w][1], iPos[w][2], box) <
                            maxRadius * maxRadius;
                        warpMask |= unsigned(jClusterOverlaps) << (warpIndex * Config::iClustersPerSupercluster + c);
                    }
                }
                warpMask = warpBitwiseOr(warpMask);

                if (warpMask)
                {
                    previousJCluster = jCluster;
                    if (info.neighborsCount >= ncmax)
                    {
                        ++info.neighborsCount;
                        continue;
                    }
                    if (laneIdx == (info.neighborsCount % GpuConfig::warpSize)) jClusterQueue = jCluster;
                    if (laneIdx == 0)
                    {
                        const unsigned maskStartIndex =
                            info.neighborsCount * (Config::iClustersPerSupercluster * Config::numWarpsPerInteraction);
                        const unsigned prevMask    = maskStartIndex % 32 == 0 ? 0 : masks[maskStartIndex / 32];
                        masks[maskStartIndex / 32] = (warpMask << (maskStartIndex % 32)) | prevMask;
                    }
                    ++info.neighborsCount;
                    if ((info.neighborsCount % GpuConfig::warpSize) == 0) compression.add(jClusterQueue, true);
                }
            }
        }
    };

    singleTraversal(tree.childOffsets, tree.parents, overlapsInternalNode, overlapsLeafNode);

    const unsigned remaining = std::min(info.neighborsCount, ncmax) % GpuConfig::warpSize;
    if (remaining != 0) compression.add(jClusterQueue, laneIdx < remaining);

    return compression.numBytes();
}

/*! compute required shared memory amount
 *
 * @param[in] ncmax maximum number of neighbor clusters
 */
template<class Config, class Tc, class ThP>
constexpr unsigned buildNbListSharedMemPerSupercluster(const unsigned ncmax)
{
    // storage requirements for uncompressed neighbor indices
    const unsigned jClustersSize = ncmax * sizeof(unsigned);
    // storage requirements for cluster-cluster interaction bitmasks
    const unsigned masksDataSize = masksSize<Config>(ncmax) * sizeof(std::uint32_t);

    return jClustersSize + masksDataSize;
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
 * @param[in]    jClusterBboxes         bounding boxes of j-clusters
 * @param[in]    nodeRMax               max. particle radii of tree nodes
 * @param[in]    ncmax                  max. number of neighbor clusters (upper bound for numCandidates)
 * @param[out]   neighborData           global memory neighbor data array where (possibly compressed) neighbor indices
 *                                      will be stored
 * @param[in]    neighborDataSize       size of neighborData array to avoid out of bounds accesses
 * @param[inout] superclusterInfo       supercluster info
 * @param[in]    numSuperClusters       number of superclusters
 * @param[in]    globalPool             global memory pool used during tree traversal
 * @param[inout] globalBuildData        global build data used to 'allocate' global memory regions per supercluster in a
 * pre-allocated array
 */
template<class Config, unsigned NumSuperclustersPerBlock, bool UsePbc, class Tc, class ThP, class KeyType>
__global__ __launch_bounds__(GpuConfig::warpSize* NumSuperclustersPerBlock) void buildNbListKernel(
    const OctreeNsView<Tc, KeyType> __grid_constant__ tree,
    const Box<Tc> __grid_constant__ box,
    const LocalIndex firstValidBody,
    const LocalIndex totalBodies,
    const LocalIndex firstBody,
    const LocalIndex lastBody,
    const Tc* const __restrict__ x,
    const Tc* const __restrict__ y,
    const Tc* const __restrict__ z,
    const ThP h,
    const JClusterBbox<Config, Tc>* const __restrict__ jClusterBboxes,
    const ThP nodeRMax,
    const unsigned ncmax,
    std::uint32_t* const __restrict__ neighborData,
    const std::size_t neighborDataSize,
    SuperclusterInfo* const __restrict__ superclusterInfo,
    const unsigned numSuperClusters,
    GlobalBuildData* __restrict__ globalBuildData)
{
    static_assert(Config::superclusterSize % GpuConfig::warpSize == 0);
    assert(blockDim.x == GpuConfig::warpSize);
    assert(blockDim.y == 1);
    assert(blockDim.z == NumSuperclustersPerBlock);

    const unsigned laneIdx = laneIndex();

    util::SharedMemAllocator sharedAllocator(buildNbListSharedMemPerSupercluster<Config, Tc, ThP>(ncmax), threadIdx.z);

    auto jClusters = sharedAllocator.alloc<std::uint32_t[]>(ncmax);
    auto masks     = sharedAllocator.alloc<std::uint32_t[]>(masksSize<Config>(ncmax));

    const unsigned firstISupercluster = superclusterIndex<Config>(firstBody);
    const unsigned lastISupercluster  = superclusterIndex<Config>(lastBody - 1) + 1;

    unsigned maxNeighbors = 0;

    while (true)
    {
        unsigned index = 0;
        if (laneIdx == 0) index = atomicAdd(&globalBuildData->index, 1);
        index = shflSync(index, 0);
        if (index >= numSuperClusters) break;

        SuperclusterInfo info = {.index = index + firstISupercluster, .neighborsCount = 0, .dataIndex = 0};

        const unsigned jClusterBytes = collectNeighborJClusters<Config, UsePbc>(
            tree, box, firstValidBody, totalBodies, x, y, z, h, jClusterBboxes, nodeRMax, ncmax, firstISupercluster,
            lastISupercluster, jClusters.get(), masks.get(), info);

        maxNeighbors = std::max(info.neighborsCount, maxNeighbors);

        if (info.neighborsCount > ncmax && laneIdx == 0) globalBuildData->status = BuildStatus::neighbor_list_overflow;

        const bool storeSuccessful = storeNeighborData<Config, NumSuperclustersPerBlock>(
            jClusters.get(), jClusterBytes, masks.get(), ncmax, neighborData, neighborDataSize,
            &globalBuildData->neighborDataSize, info);

        if (!storeSuccessful)
        {
            if (laneIdx == 0) globalBuildData->status = BuildStatus::neighbor_data_overflow;
            break;
        }

        if (laneIdx == 0) superclusterInfo[index] = info;
    }

    if (laneIdx == 0) atomicMax(&globalBuildData->maxNeighbors, maxNeighbors);
}

template<class Config, class Tc, class ThP, class KeyType>
std::size_t buildNbList(const execution::Gpu exec,
                        const OctreeNsView<Tc, KeyType>& tree,
                        const Box<Tc>& box,
                        const LocalIndex totalBodies,
                        const GroupView& groups,
                        const Tc* const x,
                        const Tc* const y,
                        const Tc* const z,
                        const ThP h,
                        const LocalIndex firstValidBody,
                        const LocalIndex numISuperclusters,
                        const JClusterBbox<Config, Tc>* const jClusterBboxes,
                        const ThP nodeRMax,
                        const unsigned ncmax,
                        std::uint32_t* const neighborData,
                        const std::size_t neighborDataVirtualSize,
                        SuperclusterInfo* const superclusterInfo)
{
    auto globalBuildData = util::deviceAlloc<GlobalBuildData>(exec);

    constexpr unsigned numSuperclustersPerBlock = 2;
    const dim3 blockSize                        = {GpuConfig::warpSize, 1, numSuperclustersPerBlock};
    const unsigned numBlocks = std::min(GpuConfig::smCount * (TravConfig::numWarpsPerSm / numSuperclustersPerBlock),
                                        (numISuperclusters + numSuperclustersPerBlock - 1) / numSuperclustersPerBlock);
    const unsigned sharedMem = numSuperclustersPerBlock * buildNbListSharedMemPerSupercluster<Config, Tc, ThP>(ncmax);

    checkGpuErrors(cudaMemsetAsync(globalBuildData.get(), 0, sizeof(GlobalBuildData), exec));

    auto run = [&](auto usePbc)
    {
        buildNbListKernel<Config, numSuperclustersPerBlock, decltype(usePbc)::value>
            <<<numBlocks, blockSize, sharedMem, exec>>>(tree, box, firstValidBody, totalBodies, groups.firstBody,
                                                        groups.lastBody, x, y, z, h, jClusterBboxes, nodeRMax, ncmax,
                                                        neighborData, neighborDataVirtualSize, superclusterInfo,
                                                        numISuperclusters, globalBuildData.get());
        checkGpuErrors(cudaGetLastError());
    };

    if (box.boundaryX() == BoundaryType::periodic || box.boundaryY() == BoundaryType::periodic ||
        box.boundaryZ() == BoundaryType::periodic)
        run(std::true_type());
    else
        run(std::false_type());

    GlobalBuildData buildData;
    checkGpuErrors(
        cudaMemcpyAsync(&buildData, globalBuildData.get(), sizeof(GlobalBuildData), cudaMemcpyDeviceToHost, exec));
    checkGpuErrors(cudaStreamSynchronize(exec));
    switch (buildData.status)
    {
        case BuildStatus::success: break;
        case BuildStatus::neighbor_list_overflow:
            std::cerr << "WARNING: overflow in cluster neighbor list in supercluster neighborhood. Missing neighbors! "
                         "Try to increase ncmax. Current ncmax is "
                      << ncmax << ", but found up to " << buildData.maxNeighbors << " neighbor clusters." << std::endl;
            break;
        case BuildStatus::neighbor_data_overflow: throw std::runtime_error("overflow in cluster neighbor data");
    }

    assert(buildData.neighborDataSize < neighborDataVirtualSize);

    return buildData.neighborDataSize;
}

} // namespace cstone::ijloop::gpu_supercluster_nb_list_neighborhood_detail
