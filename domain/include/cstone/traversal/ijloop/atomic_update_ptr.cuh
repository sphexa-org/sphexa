/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Atomic pointer updates on GPUs
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include "cstone/primitives/warpscan.cuh"
#include "cstone/traversal/ijloop/ijloop.hpp"
#include "cstone/traversal/ijloop/common.hpp"
#include "cstone/util/array.hpp"
#include "cstone/util/uninitialized.hpp"

namespace cstone::ijloop
{

namespace detail
{

template<class T>
__device__ __forceinline__ void atomicAddPtr(T* ptr, T value)
{ atomicAdd(ptr, value); }

__device__ __forceinline__ void atomicAddPtr(unsigned long* ptr, unsigned long value)
{
    static_assert(sizeof(unsigned long) == sizeof(unsigned) || sizeof(unsigned long) == sizeof(unsigned long long));
    if constexpr (sizeof(unsigned long) == sizeof(unsigned))
        atomicAdd(reinterpret_cast<unsigned*>(ptr), static_cast<unsigned>(value));
    else
        atomicAdd(reinterpret_cast<unsigned long long*>(ptr), static_cast<unsigned long long>(value));
}

template<class T, std::size_t N>
__device__ __forceinline__ void atomicAddPtr(util::array<T, N>* ptr, util::array<T, N> const& value)
{
#pragma unroll
    for (std::size_t i = 0; i < N; ++i)
        atomicAddPtr(&((*ptr)[i]), value[i]);
}

template<class T>
__device__ __forceinline__ void atomicMinPtr(T* ptr, T value)
{ atomicMin(ptr, value); }

__device__ __forceinline__ void atomicMinPtr(float* ptr, float value) { atomicMinFloat(ptr, value); }

__device__ __forceinline__ void atomicMinPtr(double* ptr, double value) { atomicMinDouble(ptr, value); }

template<class T, std::size_t N>
__device__ __forceinline__ void atomicMinPtr(util::array<T, N>* ptr, util::array<T, N> const& value)
{
#pragma unroll
    for (std::size_t i = 0; i < N; ++i)
        atomicMinPtr(&((*ptr)[i]), value[i]);
}

template<class T>
__device__ __forceinline__ void atomicMaxPtr(T* ptr, T value)
{ atomicMax(ptr, value); }

__device__ __forceinline__ void atomicMaxPtr(float* ptr, float value) { atomicMaxFloat(ptr, value); }

__device__ __forceinline__ void atomicMaxPtr(double* ptr, double value) { atomicMaxDouble(ptr, value); }

template<class T, std::size_t N>
__device__ __forceinline__ void atomicMaxPtr(util::array<T, N>* ptr, util::array<T, N> const& value)
{
#pragma unroll
    for (std::size_t i = 0; i < N; ++i)
        atomicMaxPtr(&((*ptr)[i]), value[i]);
}

} // namespace detail

template<class T>
__device__ __forceinline__ void atomicUpdatePtr(T* ptr, T const& value)
{ detail::atomicAddPtr(ptr, value); }

template<class T>
__device__ __forceinline__ void atomicUpdatePtr(T* ptr, reduction::min<T> const& value)
{ detail::atomicMinPtr(ptr, value.value); }

template<class T>
__device__ __forceinline__ void atomicUpdatePtr(T* ptr, reduction::max<T> const& value)
{ detail::atomicMaxPtr(ptr, value.value); }

template<class T, class S>
__device__ __forceinline__ void atomicUpdatePtr(T* ptr, symmetric::even<S> const& value)
{ atomicUpdatePtr(ptr, value.value); }

template<class T, class S>
__device__ __forceinline__ void atomicUpdatePtr(T* ptr, symmetric::odd<S> const& value)
{ atomicUpdatePtr(ptr, value.value); }

template<class UnwrappedReductionResult, class ReductionResult>
__device__ __forceinline__ void warpReduceUpdatePtr(UnwrappedReductionResult* const globalReductionResult,
                                                    ReductionResult reductionResult)
{
#pragma unroll
    for (unsigned offset = GpuConfig::warpSize / 2; offset >= 1; offset /= 2)
    {
        util::for_each_tuple([&](auto& value) { detail::updateResultImpl(value, shflDownSync(value, offset)); },
                             reductionResult);
    }

    if (laneIndex() == 0)
    {
        util::for_each_tuple([](auto& target, auto const& value) { atomicUpdatePtr(&target, value); },
                             *globalReductionResult, reductionResult);
    }
}

template<class UnwrappedReductionResult, class ReductionResult>
__device__ __forceinline__ void blockReduceUpdatePtr(UnwrappedReductionResult* const globalReductionResult,
                                                     ReductionResult reductionResult)
{
    __shared__ util::Uninitialized<ReductionResult> blockReductionResult;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) *blockReductionResult.data() = ReductionResult{};

    __syncthreads();

    warpReduceUpdatePtr(reinterpret_cast<UnwrappedReductionResult*>(blockReductionResult.data()), reductionResult);

    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        util::for_each_tuple([](auto& target, auto const& value) { atomicUpdatePtr(&target, value); },
                             *globalReductionResult, *blockReductionResult.data());
    }
}
} // namespace cstone::ijloop
