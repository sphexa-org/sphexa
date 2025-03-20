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
 * @brief Atomic pointer updates on GPUs
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include "cstone/primitives/warpscan.cuh"
#include "cstone/traversal/ijloop/ijloop.hpp"
#include "cstone/util/array.hpp"

namespace cstone::ijloop
{

namespace detail
{

template<class T>
__device__ __forceinline__ void atomicAddPtr(T* ptr, T value)
{
    atomicAdd(ptr, value);
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
{
    atomicMin(ptr, value);
}

__device__ __forceinline__ void atomicMinPtr(float* ptr, float value) { atomicMinFloat(ptr, value); }

template<class T, std::size_t N>
__device__ __forceinline__ void atomicMinPtr(util::array<T, N>* ptr, util::array<T, N> const& value)
{
#pragma unroll
    for (std::size_t i = 0; i < N; ++i)
        atomicMinPtr(&((*ptr)[i]), value[i]);
}

template<class T>
__device__ __forceinline__ void atomicMaxPtr(T* ptr, T value)
{
    atomicMax(ptr, value);
}

__device__ __forceinline__ void atomicMaxPtr(float* ptr, float value) { atomicMaxFloat(ptr, value); }

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
{
    detail::atomicAddPtr(ptr, value);
}

template<class T>
__device__ __forceinline__ void atomicUpdatePtr(T* ptr, reduction::min<T> const& value)
{
    detail::atomicMinPtr(ptr, value.value);
}

template<class T>
__device__ __forceinline__ void atomicUpdatePtr(T* ptr, reduction::max<T> const& value)
{
    detail::atomicMaxPtr(ptr, value.value);
}

template<class T, class S>
__device__ __forceinline__ void atomicUpdatePtr(T* ptr, symmetric::even<S> const& value)
{
    atomicUpdatePtr(ptr, value.value);
}

template<class T, class S>
__device__ __forceinline__ void atomicUpdatePtr(T* ptr, symmetric::odd<S> const& value)
{
    atomicUpdatePtr(ptr, value.value);
}

} // namespace cstone::ijloop
