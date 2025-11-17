/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Utilities for device memory handling
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <cassert>
#include <memory>
#include <type_traits>

#include "cstone/cuda/errorcheck.cuh"

namespace util
{

namespace detail
{

struct CudaFreeDeleter
{
    template<class T>
    void operator()(T* ptr) const
    {
        checkGpuErrors(cudaFree(ptr));
    }
};

} // namespace detail

template<class T>
using UniqueDevicePtr = std::unique_ptr<T, detail::CudaFreeDeleter>;

template<class T, std::enable_if_t<!std::is_array_v<T>, int> = 0>
inline UniqueDevicePtr<T> deviceAlloc()
{
    T* ptr;
    checkGpuErrors(cudaMalloc(&ptr, sizeof(T)));
    return UniqueDevicePtr<T>(ptr);
}

template<class T, std::enable_if_t<std::is_array_v<T>, int> = 0>
inline UniqueDevicePtr<T> deviceAlloc(std::size_t size)
{
    using ValueType = std::remove_extent_t<T>;
    ValueType* ptr;
    checkGpuErrors(cudaMalloc(&ptr, size * sizeof(ValueType)));
    return UniqueDevicePtr<T>(ptr);
}

} // namespace util
