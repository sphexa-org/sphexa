/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Kernels for initializing and finalizing symmetric ij-loop reductions
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/traversal/ijloop/common.hpp"

namespace cstone::ijloop
{

template<class Tc, class ThP, class In, class Out, class Interaction>
__global__ void initResultKernel(const LocalIndex firstBody,
                                 const LocalIndex lastBody,
                                 const Tc* __restrict__ x,
                                 const Tc* __restrict__ y,
                                 const Tc* __restrict__ z,
                                 const ThP h,
                                 const In input,
                                 const Out output,
                                 Interaction interaction)
{
    const LocalIndex i = blockDim.x * blockIdx.x + threadIdx.x + firstBody;
    if (i >= lastBody) return;

    using ParticleData = decltype(loadParticleData(x, y, z, h, input, firstBody));
    using Result       = decltype(interaction(ParticleData{}, ParticleData{}, Vec3<Tc>{0, 0, 0}, Tc(0)));
    storeParticleData(output, i, unwrapModifiers(Result{}));
}

template<class Config, class Tc, class ThP, class Input, class Output, class Interaction>
void initResult(const LocalIndex firstBody,
                const LocalIndex lastBody,
                const Tc* x,
                const Tc* y,
                const Tc* z,
                const ThP h,
                Input&& input,
                Output&& output,
                Interaction&& interaction)
{
    static_assert(Config::symmetric);
    const LocalIndex numBodies = lastBody - firstBody;
    constexpr unsigned threads = 256;
    const unsigned numBlocks   = iceil(numBodies, threads);
    initResultKernel<<<numBlocks, threads>>>(firstBody, lastBody, x, y, z, h, std::forward<Input>(input),
                                             std::forward<Output>(output), std::forward<Interaction>(interaction));
    checkGpuErrors(cudaGetLastError());
}

template<class Tc, class ThP, class In, class Tmp, class Out, class Postamble>
__global__ void applyPostambleKernel(const LocalIndex firstBody,
                                     const LocalIndex lastBody,
                                     const LocalIndex firstValidBody,
                                     const Tc* __restrict__ x,
                                     const Tc* __restrict__ y,
                                     const Tc* __restrict__ z,
                                     const ThP h,
                                     const In input,
                                     const Tmp tmp,
                                     const Out output,
                                     const Postamble postamble)
{
    const LocalIndex i = blockDim.x * blockIdx.x + threadIdx.x + firstBody;
    if (i >= lastBody) return;

    auto iData = loadParticleData(x, y, z, h, input, i);
    std::get<0>(iData) -= firstValidBody;
    const auto result = util::tupleMap([&](auto* ptr) { return ptr[i]; }, tmp);
    storeParticleData(output, i, postamble(iData, result));
}

template<class Config, class Tc, class ThP, class Input, class Tmp, class Output, class Postamble>
void applyPostamble(const LocalIndex firstBody,
                    const LocalIndex lastBody,
                    const LocalIndex firstValidBody,
                    const Tc* x,
                    const Tc* y,
                    const Tc* z,
                    const ThP h,
                    Input&& input,
                    Tmp&& tmp,
                    Output&& output,
                    Postamble&& postamble)
{
    static_assert(Config::symmetric);

    if constexpr (std::is_same_v<std::remove_cvref_t<Postamble>, detail::EmptyPostamble>) return;

    const LocalIndex numBodies = lastBody - firstBody;
    constexpr unsigned threads = 256;
    const unsigned numBlocks   = iceil(numBodies, threads);
    applyPostambleKernel<<<numBlocks, threads>>>(firstBody, lastBody, firstValidBody, x, y, z, h,
                                                 std::forward<Input>(input), std::forward<Tmp>(tmp),
                                                 std::forward<Output>(output), std::forward<Postamble>(postamble));
    checkGpuErrors(cudaGetLastError());
}

} // namespace cstone::ijloop
