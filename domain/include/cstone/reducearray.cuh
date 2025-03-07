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
 * @brief Fast array warp-reductions
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <tuple>

#include <cooperative_groups.h>

#include "cstone/util/array.hpp"
#include "cstone/cuda/gpu_config.cuh"

namespace cstone
{

template<unsigned ReductionSize, bool Interleave, class T, std::size_t ArraySize, class Op>
constexpr __device__ __forceinline__ T reduceArray(util::array<T, ArraySize> in, Op const& op)
{
    static_assert(ArraySize <= ReductionSize);
    const auto block              = cooperative_groups::this_thread_block();
    const auto warp               = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);
    constexpr unsigned reductions = GpuConfig::warpSize / ReductionSize;

#pragma unroll
    for (unsigned offset = 1; offset < ReductionSize; offset *= 2)
    {
#pragma unroll
        for (unsigned i = 0; i < ArraySize; i += 2 * offset)
        {
            in[i] = op(in[i], warp.shfl_down(in[i], Interleave ? offset * reductions : offset));
            if (i + offset < ArraySize)
            {
                in[i + offset] =
                    op(in[i + offset], warp.shfl_up(in[i + offset], Interleave ? offset * reductions : offset));
                const unsigned index =
                    Interleave ? warp.thread_rank() / reductions : warp.thread_rank() % ReductionSize;
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
    auto inArray = std::apply([](auto const&... args) { return util::array<T, sizeof...(Ts) + 1>{args...}; }, in);
    return reduceArray<ReductionSize, Interleave>(inArray, op);
}

} // namespace cstone
