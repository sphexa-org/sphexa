/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Tests for SFC related GPU device functions
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cstone/cuda/gpu_config.cuh"
#include "cstone/sfc/common.hpp"

using namespace cstone;

template<class T>
__global__ void nzPlaceKernel(T* values)
{
    T laneValue = threadIdx.x;

    values[threadIdx.x] = lastNzPlace(laneValue);
}

template<class T>
void nzPlaceTest()
{
    thrust::host_vector<T> h_v(GpuConfig::warpSize);
    thrust::device_vector<T> d_v = h_v;

    nzPlaceKernel<<<1, GpuConfig::warpSize>>>(thrust::raw_pointer_cast(d_v.data()));
    h_v = d_v;

    EXPECT_EQ(h_v[0], maxTreeLevel<T>{});
    EXPECT_EQ(h_v[1], maxTreeLevel<T>{});
    EXPECT_EQ(h_v[4], maxTreeLevel<T>{});
    EXPECT_EQ(h_v[8], maxTreeLevel<T>{} - 1);
}

TEST(NzPlaceGpu, lastNzPlace)
{
    nzPlaceTest<uint32_t>();
    nzPlaceTest<uint64_t>();
}