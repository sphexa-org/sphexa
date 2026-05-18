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
