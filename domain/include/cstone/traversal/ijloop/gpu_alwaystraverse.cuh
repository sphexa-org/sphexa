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
#include "cstone/traversal/ijloop/atomic_update_ptr.cuh"
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

template<bool UsePbc, class Tc, class ThP, class KeyType, class IjData>
__global__ __launch_bounds__(numThreads) void runIjLoop(
    const OctreeNsView<Tc, KeyType> __grid_constant__ tree,
    const Box<Tc> __grid_constant__ box,
    const GroupView groups,
    const Tc* __restrict__ x,
    const Tc* __restrict__ y,
    const Tc* __restrict__ z,
    const ThP h,
    const IjData ijData,
    typename IjData::UnwrappedReductionResultType* __restrict__ globalReductionResult,
    const unsigned ngmax,
    LocalIndex* __restrict__ neighbors,
    LocalIndex* __restrict__ targetCounter)
{
    const unsigned laneIdx     = threadIdx.x & (GpuConfig::warpSize - 1);
    const unsigned warpIdxGrid = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;
    LocalIndex targetIdx       = 0;

    LocalIndex* threadNeighbors = neighbors + warpIdxGrid * ngmax * GpuConfig::warpSize + laneIdx * ngmax;

    using ReductionResult = typename IjData::ReductionResultType;
    ReductionResult reductionResult{};

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

            const auto iData  = loadParticleData(x, y, z, h, makeConst(ijData.input), i);
            const bool usePbc = UsePbc && requiresPbcHandling(box, iData);
            auto result       = ijData.interaction(iData, iData, Vec3<Tc>{0, 0, 0}, Tc(0));
            for (unsigned nb = 0; nb < nbs; ++nb)
            {
                const LocalIndex j = threadNeighbors[nb];
                const auto jData   = loadParticleData(x, y, z, h, makeConst(ijData.input), j);

                const auto [ijPosDiff, distSq] = posDiffAndDistSq(usePbc, box, iData, jData);

                updateResult(result, ijData.interaction(iData, jData, ijPosDiff, distSq));
            }

            const auto postambleResult = ijData.postamble(iData, unwrapModifiers(result));
            storeParticleData(ijData.output, i, postambleResult);
            if constexpr (!std::is_same_v<typename IjData::ReductionType, detail::NoReduction>)
                updateResult(reductionResult,
                             ijData.reduction(iData, unwrapModifiers(result), unwrapModifiers(postambleResult)));
        }
    }
    if constexpr (!std::is_same_v<typename IjData::ReductionType, detail::NoReduction>)
        warpReduceUpdatePtr(globalReductionResult, reductionResult);
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

    template<class... Ts>
    auto ijLoop(const IjLoopData<Ts...>& ijData) const
    {
        return ijLoop(ijData, groups);
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

        template<class... Ts>
        auto ijLoop(const IjLoopData<Ts...>& ijData) const
        {
            return parent.ijLoop(ijData, groups);
        }
    };

    Subgroup subgroup(GroupView const& groups) const { return {*this, groups}; }

protected:
    template<class... Ts>
    auto ijLoop(const IjLoopData<Ts...>& ijData, GroupView const& groups) const
    {
        using IjLoopData               = IjLoopData<Ts...>;
        using ReductionResult          = typename IjLoopData::ReductionResultType;
        using UnwrappedReductionResult = typename IjLoopData::UnwrappedReductionResultType;
        ReductionResult reductionResult{};

        if (groups.numGroups == 0) return unwrapModifiers(reductionResult);

        util::UniqueDevicePtr<UnwrappedReductionResult> deviceReductionResult;
        if constexpr (!std::is_same_v<typename IjLoopData::ReductionType, detail::NoReduction>)
        {
            deviceReductionResult = util::deviceAlloc<UnwrappedReductionResult>(exec);
            static_assert(sizeof(ReductionResult) == sizeof(UnwrappedReductionResult));
            checkGpuErrors(cudaMemcpyAsync(deviceReductionResult.get(), &reductionResult, sizeof(ReductionResult),
                                           cudaMemcpyHostToDevice));
        }
        checkGpuErrors(cudaMemsetAsync(targetCounter.get(), 0, sizeof(LocalIndex), exec));

        if (box.boundaryX() == BoundaryType::periodic || box.boundaryY() == BoundaryType::periodic ||
            box.boundaryZ() == BoundaryType::periodic)
        {
            runIjLoop<true><<<numBlocks(), numThreads, 0, exec>>>(tree, box, groups, x, y, z, h, ijData,
                                                                  deviceReductionResult.get(), ngmax, neighbors.get(),
                                                                  targetCounter.get());
        }
        else
        {
            runIjLoop<false><<<numBlocks(), numThreads, 0, exec>>>(tree, box, groups, x, y, z, h, ijData,
                                                                   deviceReductionResult.get(), ngmax, neighbors.get(),
                                                                   targetCounter.get());
        }
        checkGpuErrors(cudaGetLastError());

        if constexpr (!std::is_same_v<typename IjLoopData::ReductionType, detail::NoReduction>)
        {
            checkGpuErrors(cudaMemcpyAsync(&reductionResult, deviceReductionResult.get(), sizeof(ReductionResult),
                                           cudaMemcpyDeviceToHost));
            checkGpuErrors(cudaStreamSynchronize(exec));
        }
        return unwrapModifiers(reductionResult);
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
