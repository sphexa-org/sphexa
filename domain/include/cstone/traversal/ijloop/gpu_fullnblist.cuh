/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Neighbor search on GPU
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <iostream>
#include <tuple>

#include <thrust/execution_policy.h>

#include "cstone/cuda/memory.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/primitives/warpscan.cuh"
#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/traversal/ijloop/common.hpp"
#include "cstone/tree/octree.hpp"

namespace cstone::ijloop
{

namespace gpu_full_nb_list_neighborhood_detail
{

template<class Tc, class ThP, class KeyType>
__global__ __launch_bounds__(TravConfig::numThreads) void gpuFullNbListNeighborhoodBuild(
    const OctreeNsView<Tc, KeyType> __grid_constant__ tree,
    const Box<Tc> __grid_constant__ box,
    const GroupView __grid_constant__ groups,
    const Tc* __restrict__ x,
    const Tc* __restrict__ y,
    const Tc* __restrict__ z,
    const ThP h,
    const unsigned ngmax,
    LocalIndex* __restrict__ neighbors,
    unsigned* __restrict__ neighborsCount,
    int* __restrict__ globalPool,
    unsigned* __restrict__ globalMaxNeighbors)
{
    const unsigned laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
    unsigned targetIdx     = 0;
    assert(groups.lastBody != 0);
    const std::size_t neighborsStride = groups.lastBody - groups.firstBody;

    neighbors -= groups.firstBody;
    neighborsCount -= groups.firstBody;

    unsigned maxNeighbors = 0;

    while (true)
    {
        if (laneIdx == 0) targetIdx = atomicAdd(&targetCounterGlob, 1);
        targetIdx = shflSync(targetIdx, 0);

        if (targetIdx >= groups.numGroups) break;

        const cstone::LocalIndex bodyBegin = groups.groupStart[targetIdx];
        const cstone::LocalIndex bodyEnd   = groups.groupEnd[targetIdx];

        std::array<unsigned, TravConfig::nwt> nc = {0};
        const auto handleInteraction             = [&](unsigned warpTarget, LocalIndex j)
        {
            const LocalIndex i = bodyBegin + warpTarget * GpuConfig::warpSize + laneIdx;
            if (nc[warpTarget] < ngmax & i < bodyEnd) neighbors[i + nc[warpTarget] * neighborsStride] = j;
            ++nc[warpTarget];
        };

        traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, handleInteraction, globalPool);

#pragma unroll
        for (unsigned warpTarget = 0; warpTarget < TravConfig::nwt; ++warpTarget)
        {
            const LocalIndex i = bodyBegin + warpTarget * GpuConfig::warpSize + laneIdx;
            if (i < bodyEnd)
            {
                neighborsCount[i] = nc[warpTarget];
                maxNeighbors      = std::max(maxNeighbors, nc[warpTarget]);
            }
        }
    }

    maxNeighbors = warpMax(maxNeighbors);
    if (laneIdx == 0) atomicMax(globalMaxNeighbors, maxNeighbors);
}

template<class Tc, class ThP, class Input, class Output, class Interaction, class Postamble>
__forceinline__ __device__ void jLoop(const Box<Tc>& box,
                                      const LocalIndex firstBody,
                                      const std::size_t neighborsStride,
                                      const Tc* __restrict__ x,
                                      const Tc* __restrict__ y,
                                      const Tc* __restrict__ z,
                                      const ThP h,
                                      Input&& input,
                                      Output&& output,
                                      Interaction&& interaction,
                                      Postamble&& postamble,
                                      const unsigned ngmax,
                                      const LocalIndex* __restrict__ neighbors,
                                      const unsigned* __restrict__ neighborsCount,
                                      const LocalIndex i)
{
    neighbors -= firstBody;
    neighborsCount -= firstBody;

    const unsigned nbs = imin(neighborsCount[i], ngmax);

    const auto iData  = loadParticleData(x, y, z, h, std::forward<Input>(input), i);
    const bool usePbc = requiresPbcHandling(box, iData);

    auto result = interaction(iData, iData, Vec3<Tc>{0, 0, 0}, Tc(0));
    for (unsigned nb = 0; nb < nbs; ++nb)
    {
        const LocalIndex j = neighbors[i + nb * neighborsStride];
        const auto jData   = loadParticleData(x, y, z, h, std::forward<Input>(input), j);

        const auto [ijPosDiff, distSq] = posDiffAndDistSq(usePbc, box, iData, jData);

        if (distSq < radiusSq(iData)) updateResult(result, interaction(iData, jData, ijPosDiff, distSq));
    }

    storeParticleData(std::forward<Output>(output), i, postamble(iData, unwrapModifiers(result)));
}

template<int MaxThreads, class Tc, class ThP, class In, class Out, class Interaction, class Postamble>
__global__ __launch_bounds__(MaxThreads) void runIjLoop(const Box<Tc> __grid_constant__ box,
                                                        const LocalIndex firstBody,
                                                        const LocalIndex lastBody,
                                                        const Tc* __restrict__ x,
                                                        const Tc* __restrict__ y,
                                                        const Tc* __restrict__ z,
                                                        const ThP h,
                                                        const In __grid_constant__ input,
                                                        const Out __grid_constant__ output,
                                                        const Interaction interaction,
                                                        const Postamble postamble,
                                                        const unsigned ngmax,
                                                        const LocalIndex* __restrict__ neighbors,
                                                        const unsigned* __restrict__ neighborsCount)
{
    const LocalIndex i = firstBody + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= lastBody) return;

    jLoop(box, firstBody, lastBody - firstBody, x, y, z, h, input, output, interaction, postamble, ngmax, neighbors,
          neighborsCount, i);
}

template<int MaxThreads, class Tc, class ThP, class In, class Out, class Interaction, class Postamble>
__global__ __launch_bounds__(MaxThreads) void runIjLoopGrouped(const Box<Tc> __grid_constant__ box,
                                                               const LocalIndex firstBody,
                                                               const LocalIndex lastBody,
                                                               const Tc* __restrict__ x,
                                                               const Tc* __restrict__ y,
                                                               const Tc* __restrict__ z,
                                                               const ThP h,
                                                               const In __grid_constant__ input,
                                                               const Out __grid_constant__ output,
                                                               const Interaction interaction,
                                                               const Postamble postamble,
                                                               const unsigned ngmax,
                                                               const LocalIndex* __restrict__ neighbors,
                                                               const unsigned* __restrict__ neighborsCount,
                                                               const GroupView __grid_constant__ groups)
{
    const unsigned laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
    const LocalIndex g     = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;
    if (g >= groups.numGroups) return;

    assert(groups.groupEnd[g] - groups.groupStart[g] <= GpuConfig::warpSize);
    const LocalIndex i = groups.groupStart[g] + laneIdx;
    if (i >= groups.groupEnd[g]) return;

    jLoop(box, firstBody, lastBody - firstBody, x, y, z, h, input, output, interaction, postamble, ngmax, neighbors,
          neighborsCount, i);
}

template<class T>
struct ScaleFunctor
{
    T factor;

    constexpr T operator()(T x) const { return x * factor; }
};

template<class Tc, class ThP>
struct GpuFullNbListNeighborhood
{
    Box<Tc> box = {0, 0};
    LocalIndex firstBody, lastBody;
    const Tc *x, *y, *z;
    ThP h;
    unsigned ngmax;
    util::UniqueDevicePtr<LocalIndex[]> neighbors;
    util::UniqueDevicePtr<unsigned[]> neighborsCount;

    template<class... In, class... Out, class Interaction, class Postamble>
    void ijLoop(std::tuple<In*...> const& input,
                std::tuple<Out*...> const& output,
                Interaction&& interaction,
                Postamble&& postamble) const
    {
        const LocalIndex numBodies = lastBody - firstBody;
        if (numBodies == 0) return;
        constexpr int numThreads = 128;
        runIjLoop<numThreads><<<iceil(numBodies, numThreads), numThreads>>>(
            box, firstBody, lastBody, x, y, z, h, makeConst(input), output, std::forward<Interaction>(interaction),
            std::forward<Postamble>(postamble), ngmax, neighbors.get(), neighborsCount.get());
        checkGpuErrors(cudaGetLastError());
    }

    Statistics stats() const
    {
        const LocalIndex numBodies = lastBody - firstBody;
        return {.numBodies = numBodies,
                .numBytes  = sizeof(LocalIndex) * numBodies * ngmax + sizeof(unsigned) * numBodies};
    }

    struct Subgroup
    {
        GpuFullNbListNeighborhood const& parent;
        GroupView groups;

        template<class... In, class... Out, class Interaction, class Postamble>
        void ijLoop(std::tuple<In*...> const& input,
                    std::tuple<Out*...> const& output,
                    Interaction&& interaction,
                    Postamble&& postamble) const
        {
            if (groups.numGroups == 0) return;
            constexpr int numThreads = 128;
            runIjLoopGrouped<numThreads><<<iceil(groups.numGroups * GpuConfig::warpSize, numThreads), numThreads>>>(
                parent.box, parent.firstBody, parent.lastBody, parent.x, parent.y, parent.z, parent.h, makeConst(input),
                output, std::forward<Interaction>(interaction), std::forward<Postamble>(postamble), parent.ngmax,
                parent.neighbors.get(), parent.neighborsCount.get(), groups);
            checkGpuErrors(cudaGetLastError());
        }
    };

    Subgroup subgroup(GroupView const& groups) const { return {*this, groups}; }
};
} // namespace gpu_full_nb_list_neighborhood_detail

struct GpuFullNbListNeighborhoodBuilder
{
    unsigned ngmax;

    template<class Tc, class KeyType, class ThP>
    gpu_full_nb_list_neighborhood_detail::GpuFullNbListNeighborhood<Tc, ThP> build(OctreeNsView<Tc, KeyType> tree,
                                                                                   const Box<Tc>& box,
                                                                                   const LocalIndex totalBodies,
                                                                                   const GroupView& groups,
                                                                                   const Tc* x,
                                                                                   const Tc* y,
                                                                                   const Tc* z,
                                                                                   const ThP h) const
    {
        using namespace gpu_full_nb_list_neighborhood_detail;
        const std::size_t numBodies = groups.lastBody - groups.firstBody;

        if (numBodies == 0) return {};

        auto neighbors      = util::deviceAlloc<LocalIndex[]>(ngmax * numBodies);
        auto neighborsCount = util::deviceAlloc<unsigned[]>(numBodies);
        auto globalPool     = util::deviceAlloc<int[]>(TravConfig::poolSize());
        auto maxNeighbors   = util::deviceAlloc<unsigned>();

        using Th = std::remove_cvref_t<std::remove_pointer_t<ThP>>;
        ThP hExt = h;
        util::UniqueDevicePtr<Th[]> hExtData;
        if (tree.searchExtFactor != 1)
        {
            if constexpr (std::is_pointer_v<ThP>)
            {
                hExtData = util::deviceAlloc<Th[]>(totalBodies);
                thrust::transform(thrust::device, h, h + totalBodies, hExtData.get(),
                                  [searchExtFactor = tree.searchExtFactor] __device__(Th hi)
                                  { return hi * searchExtFactor; });
                hExt = hExtData.get();
            }
            else { hExt = h * tree.searchExtFactor; }
            tree.searchExtFactor = 1;
        }

        checkGpuErrors(cudaMemsetAsync(maxNeighbors.get(), 0, sizeof(unsigned)));

        resetTraversalCounters<<<1, 1>>>();
        gpuFullNbListNeighborhoodBuild<<<TravConfig::numBlocks(), TravConfig::numThreads>>>(
            tree, box, groups, x, y, z, hExt, ngmax, neighbors.get(), neighborsCount.get(), globalPool.get(),
            maxNeighbors.get());
        checkGpuErrors(cudaGetLastError());

        unsigned maxNeighborsHost;
        checkGpuErrors(cudaMemcpy(&maxNeighborsHost, maxNeighbors.get(), sizeof(unsigned), cudaMemcpyDeviceToHost));
        if (maxNeighborsHost > ngmax)
        {
            std::cerr
                << "WARNING: overflow in neighbor list. Missing neighbors! Try to increase ngmax. Current ngmax is "
                << ngmax << ", but found up to " << maxNeighborsHost << " neighbor particles." << std::endl;
        }

        return {box,   groups.firstBody,     groups.lastBody,          x, y, z, h,
                ngmax, std::move(neighbors), std::move(neighborsCount)};
    }
};

} // namespace cstone::ijloop
