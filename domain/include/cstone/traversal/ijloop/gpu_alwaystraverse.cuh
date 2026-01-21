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

#include <tuple>

#include "cstone/cuda/memory.cuh"
#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/traversal/ijloop/common.hpp"
#include "cstone/tree/octree.hpp"

namespace cstone::ijloop
{

namespace gpu_always_traverse_neighborhood_detail
{

template<bool UsePbc, class Tc, class ThP, class KeyType, class In, class Out, class Interaction, class Postamble>
__global__
__launch_bounds__(TravConfig::numThreads) void runIjLoop(const OctreeNsView<Tc, KeyType> __grid_constant__ tree,
                                                         const Box<Tc> __grid_constant__ box,
                                                         const GroupView __grid_constant__ groups,
                                                         const Tc* __restrict__ x,
                                                         const Tc* __restrict__ y,
                                                         const Tc* __restrict__ z,
                                                         const ThP h,
                                                         const In __grid_constant__ input,
                                                         const Out __grid_constant__ output,
                                                         const Interaction interaction,
                                                         const Postamble postamble,
                                                         const unsigned ngmax,
                                                         LocalIndex* __restrict__ neighbors,
                                                         int* __restrict__ globalPool)
{
    const unsigned laneIdx     = threadIdx.x & (GpuConfig::warpSize - 1);
    const unsigned warpIdxGrid = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;
    unsigned targetIdx         = 0;

    unsigned* warpNidx = neighbors + warpIdxGrid * TravConfig::targetSize * ngmax;

    while (true)
    {
        if (laneIdx == 0) targetIdx = atomicAdd(&targetCounterGlob, 1);
        targetIdx = shflSync(targetIdx, 0);

        if (targetIdx >= groups.numGroups) break;

        const cstone::LocalIndex bodyBegin = groups.groupStart[targetIdx];
        const cstone::LocalIndex bodyEnd   = groups.groupEnd[targetIdx];

        auto nc_i = traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, warpNidx, ngmax, globalPool);

#pragma unroll
        for (unsigned warpTarget = 0; warpTarget < TravConfig::nwt; ++warpTarget)
        {
            const cstone::LocalIndex i = bodyBegin + warpTarget * GpuConfig::warpSize + laneIdx;
            const LocalIndex* nidx     = warpNidx + warpTarget * GpuConfig::warpSize + laneIdx;
            if (i < bodyEnd)
            {
                const auto iData = loadParticleData(x, y, z, h, input, i);

                const unsigned nbs = imin(nc_i[warpTarget], ngmax);
                auto result        = interaction(iData, iData, Vec3<Tc>{0, 0, 0}, Tc(0));
                for (unsigned nb = 0; nb < nbs; ++nb)
                {
                    const LocalIndex j             = nidx[nb * TravConfig::targetSize];
                    const auto jData               = loadParticleData(x, y, z, h, input, j);
                    const auto [ijPosDiff, distSq] = posDiffAndDistSq(UsePbc, box, iData, jData);

                    updateResult(result, interaction(iData, jData, ijPosDiff, distSq));
                }

                storeParticleData(output, i, postamble(iData, unwrapModifiers(result)));
            }
        }
    }
}

template<class Tc, class KeyType, class ThP>
struct GpuAlwaysTraverseNeighborhood
{
    OctreeNsView<Tc, KeyType> tree;
    Box<Tc> box = {0, 0};
    GroupView groups;
    const Tc *x, *y, *z;
    ThP h;
    unsigned ngmax;
    util::UniqueDevicePtr<LocalIndex[]> neighbors;
    util::UniqueDevicePtr<int[]> globalPool;

    template<class... In, class... Out, class Interaction, class Postamble>
    void ijLoop(std::tuple<In*...> const& input,
                std::tuple<Out*...> const& output,
                Interaction&& interaction,
                Postamble&& postamble) const
    {
        ijLoop(input, output, std::forward<Interaction>(interaction), std::forward<Postamble>(postamble), groups);
    }

    Statistics stats() const
    {
        return {.numBodies = groups.lastBody - groups.firstBody,
                .numBytes  = neighborsSize(ngmax) * sizeof(LocalIndex) + TravConfig::poolSize() * sizeof(int)};
    }

    static unsigned neighborsSize(unsigned ngmax)
    {
        return ngmax * TravConfig::numBlocks() * (TravConfig::numThreads / GpuConfig::warpSize) *
               TravConfig::targetSize;
    }

    struct Subgroup
    {
        GpuAlwaysTraverseNeighborhood const& parent;
        GroupView groups;

        template<class... In, class... Out, class Interaction, class Postamble>
        void ijLoop(std::tuple<In*...> const& input,
                    std::tuple<Out*...> const& output,
                    Interaction&& interaction,
                    Postamble&& postamble) const
        {
            parent.ijLoop(input, output, std::forward<Interaction>(interaction), std::forward<Postamble>(postamble),
                          groups);
        }
    };

    Subgroup subgroup(GroupView const& groups) const { return {*this, groups}; }

protected:
    template<class... In, class... Out, class Interaction, class Postamble>
    void ijLoop(std::tuple<In*...> const& input,
                std::tuple<Out*...> const& output,
                Interaction&& interaction,
                Postamble&& postamble,
                GroupView const& groups) const
    {
        if (groups.numGroups == 0) return;
        resetTraversalCounters<<<1, 1>>>();

        if (box.boundaryX() == BoundaryType::periodic | box.boundaryY() == BoundaryType::periodic |
            box.boundaryZ() == BoundaryType::periodic)
        {
            runIjLoop<true><<<TravConfig::numBlocks(), TravConfig::numThreads>>>(
                tree, box, groups, x, y, z, h, makeConst(input), output, std::forward<Interaction>(interaction),
                std::forward<Postamble>(postamble), ngmax, neighbors.get(), globalPool.get());
        }
        else
        {
            runIjLoop<false><<<TravConfig::numBlocks(), TravConfig::numThreads>>>(
                tree, box, groups, x, y, z, h, makeConst(input), output, std::forward<Interaction>(interaction),
                std::forward<Postamble>(postamble), ngmax, neighbors.get(), globalPool.get());
        }
        checkGpuErrors(cudaGetLastError());
    }
};
} // namespace gpu_always_traverse_neighborhood_detail

struct GpuAlwaysTraverseNeighborhoodBuilder
{
    unsigned ngmax;

    template<class Tc, class KeyType, class ThP>
    gpu_always_traverse_neighborhood_detail::GpuAlwaysTraverseNeighborhood<Tc, KeyType, ThP>
    build(const OctreeNsView<Tc, KeyType>& tree,
          const Box<Tc>& box,
          const LocalIndex /* totalBodies */,
          const GroupView& groups,
          const Tc* x,
          const Tc* y,
          const Tc* z,
          ThP h) const
    {
        using namespace gpu_always_traverse_neighborhood_detail;
        return {tree,
                box,
                groups,
                x,
                y,
                z,
                h,
                ngmax,
                util::deviceAlloc<LocalIndex[]>(GpuAlwaysTraverseNeighborhood<Tc, KeyType, ThP>::neighborsSize(ngmax)),
                util::deviceAlloc<int[]>(TravConfig::poolSize())};
    }
};

} // namespace cstone::ijloop
