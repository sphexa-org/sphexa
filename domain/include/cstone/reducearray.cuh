/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Fast array warp-reductions
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <array>
#include <tuple>

#include "cstone/cuda/gpu_config.cuh"
#include "cstone/primitives/warpscan.cuh"

namespace cstone
{

template<unsigned ReductionSize, bool Interleave, class T, std::size_t ArraySize, class Op>
constexpr __device__ __forceinline__ T reduceArray(std::array<T, ArraySize> in, Op const& op)
{
    static_assert(ArraySize <= ReductionSize);
    const unsigned laneIdx        = laneIndex();
    constexpr unsigned reductions = GpuConfig::warpSize / ReductionSize;

#pragma unroll
    for (unsigned offset = 1; offset < ReductionSize; offset *= 2)
    {
#pragma unroll
        for (unsigned i = 0; i < ArraySize; i += 2 * offset)
        {
            in[i] = op(in[i], shflDownSync(in[i], Interleave ? offset * reductions : offset));
            if (i + offset < ArraySize)
            {
                in[i + offset] =
                    op(in[i + offset], shflUpSync(in[i + offset], Interleave ? offset * reductions : offset));
                const unsigned index = Interleave ? laneIdx / reductions : laneIdx % ReductionSize;
                if ((index / offset) % 2) in[i] = in[i + offset];
            }
        }
    }

    return in[0];
}

template<unsigned ReductionSize, bool Interleave, class T, class... Ts, class Op>
constexpr __device__ __forceinline__ T reduceTuple(std::tuple<T, Ts...> const& in, Op const& op)
{
    static_assert(std::conjunction_v<std::is_same<T, Ts>...>);
    auto inArray = std::apply([](auto const&... args) { return std::array<T, sizeof...(Ts) + 1>{args...}; }, in);
    return reduceArray<ReductionSize, Interleave>(inArray, op);
}

} // namespace cstone
