/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Tests for GPU device functions
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cstone/cuda/gpu_config.cuh"
#include "cstone/primitives/clz.hpp"

using namespace cstone;

template<class T>
__global__ void testCtz(T* values)
{
    T laneValue = threadIdx.x;

    values[threadIdx.x] = countTrailingZeros(laneValue);
}

template<class T>
void ctzTest()
{
    thrust::host_vector<T> h_v(GpuConfig::warpSize);
    thrust::device_vector<T> d_v = h_v;

    testCtz<<<1, GpuConfig::warpSize>>>(thrust::raw_pointer_cast(d_v.data()));
    h_v = d_v;

    EXPECT_EQ(h_v[1], 0);
    EXPECT_EQ(h_v[4], 2);
    EXPECT_EQ(h_v[8], 3);
}

TEST(ClzGpu, countTrailingZeros)
{
    ctzTest<uint32_t>();
    ctzTest<uint64_t>();
}