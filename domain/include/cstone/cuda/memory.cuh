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

template<class T, std::enable_if_t<std::is_array_v<T>, int> = 0>
inline UniqueDevicePtr<T> deviceAllocVirtual(std::size_t size)
{
    using ValueType = std::remove_extent_t<T>;
    ValueType* ptr;
    // cudaMallocManaged is the easiest way to reserve a virtual address range without physical page backing or reserved
    // swap space, similar to mmap with MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE
    checkGpuErrors(cudaMallocManaged(&ptr, size * sizeof(ValueType)));
    return UniqueDevicePtr<T>(ptr);
}

struct SharedMemAllocator
{
    template<class T>
    struct SharedMemPtr
    {
        __device__ std::remove_extent_t<T>* get() { return ptr; }
        __device__ const T* get() const { return ptr; }

        __device__ std::remove_extent_t<T>& operator*() { return *ptr; }
        __device__ const T& operator*() const { return *ptr; }
        __device__ T* operator->() { return ptr; }
        __device__ const T* operator->() const { return ptr; }

        __device__ std::remove_extent_t<T>& operator[](unsigned i) { return ptr[i]; }
        __device__ const std::remove_extent_t<T>& operator[](unsigned i) const { return ptr[i]; }

        SharedMemPtr(const SharedMemPtr&) = delete;
        SharedMemPtr(SharedMemPtr&&)      = default;

        __device__ ~SharedMemPtr() { allocator.ptr -= allocSize; }

    private:
        friend struct SharedMemAllocator;

        __device__ SharedMemPtr(SharedMemAllocator& allocator, std::remove_extent_t<T>* ptr, unsigned allocSize)
            : allocator(allocator)
            , ptr(ptr)
            , allocSize(allocSize)
        {
        }

        SharedMemAllocator& allocator;
        std::remove_extent_t<T>* ptr;
        unsigned allocSize;
    };

    __device__ SharedMemAllocator(unsigned capacityPerArea = 0, unsigned areaIndex = 0)
    {
        extern __shared__ char basePtr[];
        ptr = basePtr + capacityPerArea * areaIndex;
    }

    template<class T, std::enable_if_t<!std::is_array_v<T>, int> = 0>
    __device__ SharedMemPtr<T> alloc()
    {
        auto [allocPtr, allocSize] = allocImpl<T>(1);
        return {*this, allocPtr, allocSize};
    }

    template<class T, std::enable_if_t<std::is_array_v<T>, int> = 0>
    __device__ SharedMemPtr<T> alloc(unsigned size)
    {
        auto [allocPtr, allocSize] = allocImpl<std::remove_extent_t<T>>(size);
        return {*this, allocPtr, allocSize};
    }

private:
    template<class T>
    __device__ std::tuple<T*, unsigned> allocImpl(unsigned size)
    {

        unsigned offset    = (alignof(T) - reinterpret_cast<std::size_t>(ptr)) % alignof(T);
        unsigned allocSize = size * sizeof(T) + offset;
        T* allocated       = reinterpret_cast<T*>(ptr + offset);
        ptr += allocSize;
        return {allocated, allocSize};
    }

    char* ptr;
};

} // namespace util
