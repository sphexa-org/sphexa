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

#include "cstone/execution.hpp"
#include "cstone/cuda/memory.cuh"
#include "cstone/findneighbors.hpp"
#include "cstone/primitives/warpscan.cuh"
#include "cstone/traversal/ijloop/common.hpp"
#include "cstone/tree/octree.hpp"

namespace cstone::ijloop
{

namespace gpu_always_traverse_neighborhood_detail
{

constexpr unsigned numThreads = 128;

inline unsigned numBlocks()
{
    constexpr unsigned numWarpsPerSm = 40;
    return GpuConfig::smCount * (numWarpsPerSm / (numThreads / GpuConfig::warpSize));
}

template<bool UsePbc, class Tc, class ThP, class KeyType, class Input, class Output, class Interaction, class Postamble>
__global__ __launch_bounds__(numThreads) void runIjLoop(const OctreeNsView<Tc, KeyType> __grid_constant__ tree,
                                                        const Box<Tc> __grid_constant__ box,
                                                        const GroupView groups,
                                                        const Tc* __restrict__ x,
                                                        const Tc* __restrict__ y,
                                                        const Tc* __restrict__ z,
                                                        const ThP h,
                                                        const Input input,
                                                        const Output output,
                                                        const Interaction interaction,
                                                        const Postamble postamble,
                                                        const unsigned ngmax,
                                                        LocalIndex* __restrict__ neighbors,
                                                        LocalIndex* __restrict__ targetCounter)
{
    const unsigned laneIdx     = threadIdx.x & (GpuConfig::warpSize - 1);
    const unsigned warpIdxGrid = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;
    LocalIndex targetIdx       = 0;

    LocalIndex* threadNeighbors = neighbors + warpIdxGrid * ngmax * GpuConfig::warpSize + laneIdx * ngmax;

    while (true)
    {
        if (laneIdx == 0) targetIdx = atomicAdd(targetCounter, 1);
        targetIdx = shflSync(targetIdx, 0);

        if (targetIdx >= groups.numGroups) break;

        const LocalIndex bodyBegin = groups.groupStart[targetIdx];
        const LocalIndex bodyEnd   = groups.groupEnd[targetIdx];

        const LocalIndex i = bodyBegin + laneIdx;
        if (i < bodyEnd)
        {
            const unsigned nbs = std::min(findNeighbors(i, x, y, z, h, tree, box, ngmax, threadNeighbors), ngmax);

            const auto iData = loadParticleData(x, y, z, h, input, i);
            if (!hasInfiniteH(iData))
            {
                const bool usePbc = UsePbc && requiresPbcHandling(box, iData);
                auto result       = interaction(iData, iData, Vec3<Tc>{0, 0, 0}, Tc(0));
                for (unsigned nb = 0; nb < nbs; ++nb)
                {
                    const LocalIndex j = threadNeighbors[nb];
                    const auto jData   = loadParticleData(x, y, z, h, input, j);
                    if (hasInfiniteH(jData)) continue;

                    const auto [ijPosDiff, distSq] = posDiffAndDistSq(usePbc, box, iData, jData);

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
    execution::Gpu exec = execution::gpuDefaultStream;
    OctreeNsView<Tc, KeyType> tree;
    Box<Tc> box = {0, 0};
    GroupView groups;
    const Tc *x, *y, *z;
    ThP h;
    unsigned ngmax;
    util::UniqueDevicePtr<LocalIndex[]> neighbors;
    util::UniqueDevicePtr<LocalIndex> targetCounter;

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
        return {.numBodies = groups.lastBody - groups.firstBody, .numBytes = neighborsSize(ngmax) * sizeof(LocalIndex)};
    }

    static unsigned neighborsSize(unsigned ngmax) { return ngmax * numBlocks() * numThreads; }

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
        checkGpuErrors(cudaMemsetAsync(targetCounter.get(), 0, sizeof(LocalIndex), exec));

        if (box.boundaryX() == BoundaryType::periodic || box.boundaryY() == BoundaryType::periodic ||
            box.boundaryZ() == BoundaryType::periodic)
        {
            runIjLoop<true><<<numBlocks(), numThreads, 0, exec>>>(
                tree, box, groups, x, y, z, h, makeConst(input), output, std::forward<Interaction>(interaction),
                std::forward<Postamble>(postamble), ngmax, neighbors.get(), targetCounter.get());
        }
        else
        {
            runIjLoop<false><<<numBlocks(), numThreads, 0, exec>>>(
                tree, box, groups, x, y, z, h, makeConst(input), output, std::forward<Interaction>(interaction),
                std::forward<Postamble>(postamble), ngmax, neighbors.get(), targetCounter.get());
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
    build(execution::Gpu exec,
          const OctreeNsView<Tc, KeyType>& tree,
          const Box<Tc>& box,
          const LocalIndex /* totalBodies */,
          const GroupView& groups,
          const Tc* x,
          const Tc* y,
          const Tc* z,
          ThP h) const
    {
        using namespace gpu_always_traverse_neighborhood_detail;
        return {exec,
                tree,
                box,
                groups,
                x,
                y,
                z,
                h,
                ngmax,
                util::deviceAlloc<LocalIndex[]>(exec,
                                                GpuAlwaysTraverseNeighborhood<Tc, KeyType, ThP>::neighborsSize(ngmax)),
                util::deviceAlloc<LocalIndex>(exec)};
    }
};

} // namespace cstone::ijloop
