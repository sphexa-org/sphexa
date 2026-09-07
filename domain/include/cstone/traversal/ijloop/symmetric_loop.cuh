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
#include "cstone/execution.hpp"
#include "cstone/primitives/math.hpp"
#include "cstone/traversal/ijloop/common.hpp"
#include "cstone/traversal/ijloop/atomic_update_ptr.cuh"

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
void initResult(execution::Gpu exec,
                const LocalIndex firstBody,
                const LocalIndex lastBody,
                const Tc* x,
                const Tc* y,
                const Tc* z,
                const ThP h,
                Input const& input,
                Output const& output,
                Interaction&& interaction)
{
    static_assert(Config::symmetric);
    const LocalIndex numBodies = lastBody - firstBody;
    constexpr unsigned threads = 256;
    const unsigned numBlocks   = iceil(numBodies, threads);
    initResultKernel<<<numBlocks, threads, 0, exec>>>(firstBody, lastBody, x, y, z, h, input, output, interaction);
    checkGpuErrors(cudaGetLastError());
}

template<class Tc,
         class ThP,
         class In,
         class Tmp,
         class Out,
         class Postamble,
         class Reduction,
         class UnwrappedReductionResult>
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
                                     const Postamble postamble,
                                     const Reduction reduction,
                                     UnwrappedReductionResult* const __restrict__ globalReductionResult)
{
    const LocalIndex i    = blockDim.x * blockIdx.x + threadIdx.x + firstBody;
    using ParticleData    = decltype(loadParticleData(x, y, z, h, input, i));
    using Result          = decltype(util::tupleMap([&](auto* ptr) { return ptr[i]; }, tmp));
    using ReductionResult = std::decay_t<decltype(reduction(
        std::declval<ParticleData>(), unwrapModifiers(std::declval<Result>()),
        unwrapModifiers(postamble(std::declval<ParticleData>(), std::declval<Result>()))))>;
    ReductionResult reductionResult{};
    if (i < lastBody)
    {
        auto iData = loadParticleData(x, y, z, h, input, i);
        std::get<0>(iData) -= firstValidBody;
        const auto result          = util::tupleMap([&](auto* ptr) { return ptr[i]; }, tmp);
        const auto postambleResult = postamble(iData, result);
        storeParticleData(output, i, postambleResult);

        if constexpr (!std::is_same_v<Reduction, detail::NoReduction>)
            reductionResult = reduction(iData, unwrapModifiers(result), unwrapModifiers(postambleResult));
    }
    if constexpr (!std::is_same_v<Reduction, detail::NoReduction>)
        blockReduceUpdatePtr(globalReductionResult, reductionResult);
}

template<class Config,
         class Tc,
         class ThP,
         class Input,
         class Tmp,
         class Output,
         class Postamble,
         class Reduction,
         class UnwrappedReductionResult>
void applyPostamble(execution::Gpu exec,
                    const LocalIndex firstBody,
                    const LocalIndex lastBody,
                    const LocalIndex firstValidBody,
                    const Tc* x,
                    const Tc* y,
                    const Tc* z,
                    const ThP h,
                    Input const& input,
                    Tmp const& tmp,
                    Output const& output,
                    Postamble const& postamble,
                    Reduction const& reduction,
                    UnwrappedReductionResult* reductionResult)
{
    static_assert(Config::symmetric);

    if constexpr (std::is_same_v<Postamble, detail::EmptyPostamble> && std::is_same_v<Reduction, detail::NoReduction>)
        return;

    const LocalIndex numBodies = lastBody - firstBody;
    constexpr unsigned threads = 256;
    const unsigned numBlocks   = iceil(numBodies, threads);
    applyPostambleKernel<<<numBlocks, threads, 0, exec>>>(firstBody, lastBody, firstValidBody, x, y, z, h, input, tmp,
                                                          output, postamble, reduction, reductionResult);
    checkGpuErrors(cudaGetLastError());
}

} // namespace cstone::ijloop
