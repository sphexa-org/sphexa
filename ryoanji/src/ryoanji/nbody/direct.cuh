/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief All-to-all direct sum implementation of gravitational forces
 *
 * Inspired by
 * https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "kahan.hpp"
#include "kernel.hpp"
#include "types.h"

namespace ryoanji
{

template<class T>
using kvec4 = util::array<kahan<T>, 4>;

struct DirectConfig
{
    static constexpr int numThreads = 256;
};

template<class T>
__global__ void directKernel(unsigned first, unsigned last, unsigned numSource, Vec3<T> pbcShift,
                             const T* __restrict__ x, const T* __restrict__ y, const T* __restrict__ z,
                             const T* __restrict__ m, const T* __restrict__ h, T* p, T* ax, T* ay, T* az)
{
    unsigned targetIdx = first + blockDim.x * blockIdx.x + threadIdx.x;

    Vec3<T> pos_i = {T(0), T(0), T(0)};
    T       h_i   = 1.0;
    if (targetIdx < last)
    {
        pos_i = Vec3<T>{x[targetIdx], y[targetIdx], z[targetIdx]} + pbcShift;
        h_i   = h[targetIdx];
    }

    // kvec4<T> acc = {0.0, 0.0, 0.0, 0.0};
    util::array<T, 4> acc{0, 0, 0, 0};

    __shared__ util::array<T, 5> sm_bodytile[DirectConfig::numThreads];
    unsigned                     numTiles = (numSource - 1) / blockDim.x + 1;
    for (unsigned tile = 0; tile < numTiles; ++tile)
    {
        unsigned sourceIdx = tile * blockDim.x + threadIdx.x;
        if (sourceIdx < numSource)
            sm_bodytile[threadIdx.x] = {x[sourceIdx], y[sourceIdx], z[sourceIdx], m[sourceIdx], h[sourceIdx]};
        else
            sm_bodytile[threadIdx.x] = {0, 0, 0, 0, 1.0};

        __syncthreads();

        for (int j = 0; j < blockDim.x; ++j)
        {
            Vec3<T> pos_j{sm_bodytile[j][0], sm_bodytile[j][1], sm_bodytile[j][2]};
            T       q_j = sm_bodytile[j][3];
            T       h_j = sm_bodytile[j][4];

            acc = P2P(acc, pos_i, pos_j, q_j, h_i, h_j);
        }

        __syncthreads();
    }

    if (targetIdx < last)
    {
        p[targetIdx] += m[targetIdx] * T(acc[0]);
        ax[targetIdx] += T(acc[1]);
        ay[targetIdx] += T(acc[2]);
        az[targetIdx] += T(acc[3]);
    }
}

template<class T>
void directSum(size_t first, size_t last, size_t numBodies, Vec3<T> boxL, int numShells, const T* x, const T* y,
               const T* z, const T* m, const T* h, T* p, T* ax, T* ay, T* az)
{
    size_t   numTargets = last - first;
    unsigned numThreads = DirectConfig::numThreads;
    unsigned numBlocks  = (numTargets - 1) / numThreads + 1;

    for (int iz = -numShells; iz <= numShells; ++iz)
    {
        for (int iy = -numShells; iy <= numShells; ++iy)
        {
            for (int ix = -numShells; ix <= numShells; ++ix)
            {
                auto pbcShift = Vec3<T>{ix * boxL[0], iy * boxL[1], iz * boxL[2]};
                directKernel<<<numBlocks, numThreads>>>(first, last, numBodies, pbcShift, x, y, z, m, h, p, ax, ay, az);
                kernelSuccess("direct sum");
            }
        }
    }
}

} // namespace ryoanji
