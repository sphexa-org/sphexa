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
#include "cstone/execution.hpp"

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

struct CudaFreeAsyncDeleter
{
    cudaStream_t stream;

    template<class T>
    void operator()(T* ptr) const
    {
        checkGpuErrors(cudaFreeAsync(ptr, stream));
    }
};

} // namespace detail

template<class T>
using UniqueDevicePtr = std::unique_ptr<T, detail::CudaFreeAsyncDeleter>;

template<class T>
using UniqueManagedPtr = std::unique_ptr<T, detail::CudaFreeDeleter>;

template<class T, std::enable_if_t<!std::is_array_v<T>, int> = 0>
inline UniqueDevicePtr<T> deviceAlloc(cstone::execution::Gpu exec)
{
    T* ptr;
    checkGpuErrors(cudaMallocAsync(&ptr, sizeof(T), exec));
    return UniqueDevicePtr<T>(ptr, detail::CudaFreeAsyncDeleter{exec});
}

template<class T, std::enable_if_t<std::is_array_v<T>, int> = 0>
inline UniqueDevicePtr<T> deviceAlloc(cstone::execution::Gpu exec, std::size_t size)
{
    using ValueType = std::remove_extent_t<T>;
    ValueType* ptr;
    checkGpuErrors(cudaMallocAsync(&ptr, size * sizeof(ValueType), exec));
    return UniqueDevicePtr<T>(ptr, detail::CudaFreeAsyncDeleter{exec});
}

template<class T, std::enable_if_t<std::is_array_v<T>, int> = 0>
inline UniqueManagedPtr<T> deviceAllocVirtual(std::size_t size)
{
    using ValueType = std::remove_extent_t<T>;
    ValueType* ptr;
    // cudaMallocManaged is the easiest way to reserve a virtual address range without physical page backing or reserved
    // swap space, similar to mmap with MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE
    checkGpuErrors(cudaMallocManaged(&ptr, size * sizeof(ValueType)));
    return UniqueManagedPtr<T>(ptr);
}

struct SharedMemAllocator
{
    template<class T>
    struct SharedMemPtr
    {
        __device__ constexpr std::remove_extent_t<T>* get()
        {
            assert(ptr);
            return ptr;
        }
        __device__ constexpr const std::remove_extent_t<T>* get() const
        {
            assert(ptr);
            return ptr;
        }

        __device__ constexpr std::remove_extent_t<T>& operator*() { return *get(); }
        __device__ constexpr const std::remove_extent_t<T>& operator*() const { return *get(); }
        __device__ constexpr std::remove_extent_t<T>* operator->() { return get(); }
        __device__ constexpr const std::remove_extent_t<T>* operator->() const { return get(); }

        __device__ constexpr std::remove_extent_t<T>& operator[](unsigned i) { return get()[i]; }
        __device__ constexpr const std::remove_extent_t<T>& operator[](unsigned i) const { return get()[i]; }

        constexpr SharedMemPtr(const SharedMemPtr&) = delete;
        constexpr SharedMemPtr(SharedMemPtr&& other)
            : allocator(other.allocator)
            , ptr(other.ptr)
            , allocSize(other.allocSize)
        {
            other.ptr       = nullptr;
            other.allocSize = 0;
        }

        __device__ constexpr ~SharedMemPtr()
        {
            allocator.ptr -= allocSize;
            allocator.size -= allocSize;
        }

    private:
        friend struct SharedMemAllocator;

        __device__ constexpr SharedMemPtr(SharedMemAllocator& allocator,
                                          std::remove_extent_t<T>* ptr,
                                          unsigned allocSize)
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
        ptr      = basePtr + capacityPerArea * areaIndex;
        size     = 0;
        capacity = capacityPerArea;
    }

    constexpr SharedMemAllocator(SharedMemAllocator const&) = delete;
    constexpr SharedMemAllocator(SharedMemAllocator&&)      = default;

    template<class T, std::enable_if_t<!std::is_array_v<T>, int> = 0>
    __device__ constexpr SharedMemPtr<T> alloc()
    {
        auto [allocPtr, allocSize] = allocImpl<T>(1);
        return {*this, allocPtr, allocSize};
    }

    template<class T, std::enable_if_t<std::is_array_v<T>, int> = 0>
    __device__ constexpr SharedMemPtr<T> alloc(unsigned size)
    {
        auto [allocPtr, allocSize] = allocImpl<std::remove_extent_t<T>>(size);
        return {*this, allocPtr, allocSize};
    }

private:
    template<class T>
    __device__ constexpr std::tuple<T*, unsigned> allocImpl(unsigned n)
    {

        unsigned offset    = (alignof(T) - reinterpret_cast<std::size_t>(ptr)) % alignof(T);
        unsigned allocSize = n * sizeof(T) + offset;
        T* allocated       = reinterpret_cast<T*>(ptr + offset);
        ptr += allocSize;
        size += allocSize;
        assert(size <= capacity);
        return {allocated, allocSize};
    }

    char* ptr;
    unsigned size, capacity;
};

} // namespace util
