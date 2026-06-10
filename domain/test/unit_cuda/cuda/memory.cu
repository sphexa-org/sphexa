/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Tests for GPU memory utilities
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#include "gtest/gtest.h"

#include "cstone/cuda/gpu_config.cuh"
#include "cstone/cuda/memory.cuh"

using namespace cstone;

template<class T>
__global__ void deviceAccess(T* ptr, T value)
{
    ptr[threadIdx.x] = value;
}

TEST(Memory, DeviceAllocScalar)
{
    auto data = util::deviceAlloc<int>();
    ASSERT_TRUE(data);
    deviceAccess<<<1, 1>>>(data.get(), 42);
    checkGpuErrors(cudaDeviceSynchronize());
}

TEST(Memory, DeviceAllocArray)
{
    auto data = util::deviceAlloc<float[]>(10);
    ASSERT_TRUE(data);
    deviceAccess<<<1, 10>>>(data.get(), 37.5f);
    checkGpuErrors(cudaDeviceSynchronize());
}

TEST(Memory, DeviceAllocVirtual)
{
    // allocate 64 TiB of data, makes sure we only allocate virtual memory, not physical
    auto data = util::deviceAllocVirtual<float[]>(1024ul * 1024ul * 1024ul * 1024ul * 16ul);
    ASSERT_TRUE(data);
    deviceAccess<<<1, 10>>>(data.get(), 37.5f);
    checkGpuErrors(cudaDeviceSynchronize());
}

__global__ void testSharedMemAlloc(bool* failed)
{
    util::SharedMemAllocator alloc((1 + GpuConfig::warpSize) * sizeof(double), threadIdx.x / GpuConfig::warpSize);
    auto charData     = alloc.alloc<char[]>(3);
    const int laneIdx = threadIdx.x % GpuConfig::warpSize;
    if (laneIdx < 3) charData[laneIdx] = 'a' + threadIdx.x % 5;

    __syncthreads();

    void* ptr;
    {
        auto doubleData     = alloc.alloc<double[]>(GpuConfig::warpSize);
        doubleData[laneIdx] = threadIdx.x;
        ptr                 = doubleData.get();

        __syncthreads();

        if (doubleData[laneIdx] != threadIdx.x)
        {
            printf("doubleData check failed\n");
            *failed = true;
        }
        auto movedDoubleData = std::move(doubleData);
        if (movedDoubleData[laneIdx] != threadIdx.x)
        {
            printf("movedDoubleData check failed\n");
            *failed = true;
        }
    }

    {
        auto i64Data = alloc.alloc<std::int64_t[]>(GpuConfig::warpSize);
        if (i64Data.get() != ptr)
        {
            printf("i64Data check failed\n");
            *failed = true;
        }
        auto movedI64Data = std::move(i64Data);
        if (movedI64Data.get() != ptr)
        {
            printf("movedI64Data check failed\n");
            *failed = true;
        }
    }

    if (laneIdx < 3 && charData[laneIdx] != 'a' + threadIdx.x % 5)
    {
        printf("charData check failed\n");
        *failed = true;
    }
}

TEST(Memory, SharedMemAllocator)
{
    auto failedDevice = util::deviceAlloc<bool>();
    checkGpuErrors(cudaMemset(failedDevice.get(), 0, sizeof(bool)));
    testSharedMemAlloc<<<1, 2 * GpuConfig::warpSize, (1 + GpuConfig::warpSize) * sizeof(double) * 2>>>(
        failedDevice.get());
    checkGpuErrors(cudaDeviceSynchronize());
    bool failedHost;
    checkGpuErrors(cudaMemcpy(&failedHost, failedDevice.get(), sizeof(bool), cudaMemcpyDeviceToHost));
    ASSERT_FALSE(failedHost);
}
