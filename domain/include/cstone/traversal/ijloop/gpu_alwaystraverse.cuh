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

#include <tuple>

#include "cstone/cuda/thrust_util.cuh"
#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/traversal/ijloop/ijloop.hpp"
#include "cstone/tree/octree.hpp"

namespace cstone::ijloop
{

namespace detail
{

template<bool UsePbc, class Tc, class Th, class KeyType, class In, class Out, class Interaction>
__global__ __launch_bounds__(TravConfig::numThreads) void gpuAlwaysTraverseNeighborhoodKernel(
    const OctreeNsView<Tc, KeyType> __grid_constant__ tree,
    const Box<Tc> __grid_constant__ box,
    const GroupView __grid_constant__ groups,
    const Tc* __restrict__ x,
    const Tc* __restrict__ y,
    const Tc* __restrict__ z,
    const Th* __restrict__ h,
    const In __grid_constant__ input,
    const Out __grid_constant__ output,
    const Interaction interaction,
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

                storeParticleData(output, i, result);
            }
        }
    }
}

template<class Tc, class KeyType, class Th>
struct GpuAlwaysTraverseNeighborhoodImpl
{
    OctreeNsView<Tc, KeyType> tree;
    Box<Tc> box;
    GroupView groups;
    const Tc *x, *y, *z;
    const Th* h;
    unsigned ngmax;
    mutable thrust::device_vector<LocalIndex> neighbors;
    mutable thrust::device_vector<int> globalPool;

    template<class... In, class... Out, class Interaction, Symmetry Sym>
    void
    ijLoop(std::tuple<In*...> const& input, std::tuple<Out*...> const& output, Interaction&& interaction, Sym) const
    {
        resetTraversalCounters<<<1, 1>>>();
        if (box.boundaryX() == BoundaryType::periodic | box.boundaryY() == BoundaryType::periodic |
            box.boundaryZ() == BoundaryType::periodic)
        {
            gpuAlwaysTraverseNeighborhoodKernel<true><<<TravConfig::numBlocks(), TravConfig::numThreads>>>(
                tree, box, groups, x, y, z, h, makeConstRestrict(input), output, std::forward<Interaction>(interaction),
                ngmax, rawPtr(neighbors), rawPtr(globalPool));
        }
        else
        {
            gpuAlwaysTraverseNeighborhoodKernel<false><<<TravConfig::numBlocks(), TravConfig::numThreads>>>(
                tree, box, groups, x, y, z, h, makeConstRestrict(input), output, std::forward<Interaction>(interaction),
                ngmax, rawPtr(neighbors), rawPtr(globalPool));
        }
        checkGpuErrors(cudaGetLastError());
    }

    Statistics stats() const
    {
        return {.numBodies = groups.lastBody - groups.firstBody,
                .numBytes  = neighbors.size() * sizeof(typename decltype(neighbors)::value_type) +
                            globalPool.size() * sizeof(typename decltype(globalPool)::value_type)};
    }
};
} // namespace detail

struct GpuAlwaysTraverseNeighborhood
{
    unsigned ngmax;

    template<class Tc, class KeyType, class Th>
    detail::GpuAlwaysTraverseNeighborhoodImpl<Tc, KeyType, Th> build(const OctreeNsView<Tc, KeyType>& tree,
                                                                     const Box<Tc>& box,
                                                                     const LocalIndex /* totalBodies */,
                                                                     const GroupView& groups,
                                                                     const Tc* x,
                                                                     const Tc* y,
                                                                     const Tc* z,
                                                                     const Th* h) const
    {
        return {tree,
                box,
                groups,
                x,
                y,
                z,
                h,
                ngmax,
                thrust::device_vector<LocalIndex>(ngmax * TravConfig::numBlocks() *
                                                  (TravConfig::numThreads / GpuConfig::warpSize) *
                                                  TravConfig::targetSize),
                thrust::device_vector<int>(TravConfig::poolSize())};
    }
};

} // namespace cstone::ijloop
