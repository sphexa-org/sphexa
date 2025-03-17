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

template<class T>
__device__ __forceinline__ void atomicUpdatePtr(T* ptr, T value)
{
    atomicAdd(ptr, value);
}

template<class T>
__device__ __forceinline__ void atomicUpdatePtr(T* ptr, reduction::min<T> value)
{
    atomicMin(ptr, value.value);
}

__device__ __forceinline__ void atomicUpdatePtr(float* ptr, reduction::min<float> value)
{
    atomicMinFloat(ptr, value.value);
}

template<class T>
__device__ __forceinline__ void atomicUpdatePtr(reduction::min<T>* ptr, reduction::min<T> value)
{
    atomicUpdatePtr(&ptr->value, value);
}

template<class T>
__device__ __forceinline__ void atomicUpdatePtr(T* ptr, reduction::max<T> value)
{
    atomicMax(ptr, value.value);
}

__device__ __forceinline__ void atomicUpdatePtr(float* ptr, reduction::max<float> value)
{
    atomicMaxFloat(ptr, value.value);
}

template<class T>
__device__ __forceinline__ void atomicUpdatePtr(reduction::max<T>* ptr, reduction::max<T> value)
{
    atomicUpdatePtr(&ptr->value, value);
}

template<class T, std::size_t N>
__device__ __forceinline__ void atomicUpdatePtr(util::array<T, N>* ptr, util::array<T, N> const& value)
{
#pragma unroll
    for (std::size_t i = 0; i < N; ++i)
        atomicUpdatePtr(&((*ptr)[i]), value[i]);
}

} // namespace cstone::ijloop
