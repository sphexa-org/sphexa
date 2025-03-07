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
#include "cstone/primitives/math.hpp"
#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/traversal/ijloop/ijloop.hpp"
#include "cstone/tree/octree.hpp"

namespace cstone::ijloop
{

namespace detail
{

template<int MaxThreads, class Tc, class Th, class KeyType>
__global__ __launch_bounds__(MaxThreads) void gpuFullNbListNeighborhoodBuild(
    const OctreeNsView<Tc, KeyType> __grid_constant__ tree,
    const Box<Tc> __grid_constant__ box,
    const LocalIndex firstBody,
    const LocalIndex lastBody,
    const Tc* __restrict__ x,
    const Tc* __restrict__ y,
    const Tc* __restrict__ z,
    const Th* __restrict__ h,
    const unsigned ngmax,
    LocalIndex* neighbors,
    unsigned* neighborsCount)
{
    const LocalIndex threadId = blockDim.x * blockIdx.x + threadIdx.x;
    const LocalIndex i        = firstBody + threadId;
    if (i >= lastBody) return;

    const std::size_t neighborsStride = lastBody - firstBody;
    neighborsCount[threadId] = findNeighbors(i, x, y, z, h, tree, box, ngmax, neighbors + threadId, neighborsStride);
}

template<int MaxThreads, class Tc, class Th, class In, class Out, class Interaction>
__global__
__launch_bounds__(MaxThreads) void gpuFullNbListNeighborhoodKernel(const Box<Tc> __grid_constant__ box,
                                                                   const LocalIndex firstBody,
                                                                   const LocalIndex lastBody,
                                                                   const Tc* __restrict__ x,
                                                                   const Tc* __restrict__ y,
                                                                   const Tc* __restrict__ z,
                                                                   const Th* __restrict__ h,
                                                                   const In __grid_constant__ input,
                                                                   const Out __grid_constant__ output,
                                                                   const Interaction interaction,
                                                                   const unsigned ngmax,
                                                                   const LocalIndex* __restrict__ neighbors,
                                                                   const unsigned* __restrict__ neighborsCount)
{
    const LocalIndex threadId = blockDim.x * blockIdx.x + threadIdx.x;
    const LocalIndex i        = firstBody + threadId;
    if (i >= lastBody) return;

    const std::size_t neighborsStride = lastBody - firstBody;
    const unsigned nbs                = imin(neighborsCount[threadId], ngmax);

    const auto iData  = loadParticleData(x, y, z, h, input, i);
    const bool usePbc = requiresPbcHandling(box, iData);

    auto result = interaction(iData, iData, Vec3<Tc>{0, 0, 0}, Tc(0));
    for (unsigned nb = 0; nb < nbs; ++nb)
    {
        const LocalIndex j = neighbors[threadId + nb * neighborsStride];
        const auto jData   = loadParticleData(x, y, z, h, input, j);

        const auto [ijPosDiff, distSq] = posDiffAndDistSq(usePbc, box, iData, jData);

        updateResult(result, interaction(iData, jData, ijPosDiff, distSq));
    }

    storeParticleData(output, i, result);
}

template<class Tc, class Th>
struct GpuFullNbListNeighborhoodImpl
{
    Box<Tc> box;
    LocalIndex firstBody, lastBody;
    const Tc *x, *y, *z;
    const Th* h;
    unsigned ngmax;
    thrust::device_vector<LocalIndex> neighbors;
    thrust::device_vector<unsigned> neighborsCount;

    template<class... In, class... Out, class Interaction, Symmetry Sym>
    void
    ijLoop(std::tuple<In*...> const& input, std::tuple<Out*...> const& output, Interaction&& interaction, Sym) const
    {
        const LocalIndex numBodies = lastBody - firstBody;
        constexpr int numThreads   = 128;
        detail::gpuFullNbListNeighborhoodKernel<numThreads><<<iceil(numBodies, numThreads), numThreads>>>(
            box, firstBody, lastBody, x, y, z, h, makeConstRestrict(input), output,
            std::forward<Interaction>(interaction), ngmax, rawPtr(neighbors), rawPtr(neighborsCount));
        checkGpuErrors(cudaGetLastError());
    }

    Statistics stats() const
    {
        return {.numBodies = lastBody - firstBody,
                .numBytes  = neighbors.size() * sizeof(typename decltype(neighbors)::value_type) +
                            neighborsCount.size() * sizeof(typename decltype(neighborsCount)::value_type)};
    }
};
} // namespace detail

struct GpuFullNbListNeighborhood
{
    unsigned ngmax;

    template<class Tc, class KeyType, class Th>
    detail::GpuFullNbListNeighborhoodImpl<Tc, Th> build(const OctreeNsView<Tc, KeyType>& tree,
                                                        const Box<Tc>& box,
                                                        const LocalIndex /*totalBodies*/,
                                                        const GroupView& groups,
                                                        const Tc* x,
                                                        const Tc* y,
                                                        const Tc* z,
                                                        const Th* h) const
    {
        const LocalIndex numBodies = groups.lastBody - groups.firstBody;
        detail::GpuFullNbListNeighborhoodImpl<Tc, Th> nbList{
            box,
            groups.firstBody,
            groups.lastBody,
            x,
            y,
            z,
            h,
            ngmax,
            thrust::device_vector<LocalIndex>(ngmax * std::size_t(numBodies)),
            thrust::device_vector<int>(numBodies)};
        constexpr int numThreads = 128;
        detail::gpuFullNbListNeighborhoodBuild<numThreads><<<iceil(numBodies, numThreads), numThreads>>>(
            tree, box, groups.firstBody, groups.lastBody, x, y, z, h, ngmax, rawPtr(nbList.neighbors),
            rawPtr(nbList.neighborsCount));
        checkGpuErrors(cudaGetLastError());
        return nbList;
    }
};

} // namespace cstone::ijloop
