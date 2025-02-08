/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Tests for warp-level primitives
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cstone/cuda/thrust_util.cuh"
#include "cstone/primitives/warpscan.cuh"

using namespace cstone;

__global__ void testMin(int* values)
{
    int laneValue = threadIdx.x;

    values[threadIdx.x] = warpMin(laneValue);
}

TEST(WarpScan, min)
{
    thrust::host_vector<int> h_v(GpuConfig::warpSize);
    thrust::device_vector<int> d_v = h_v;

    testMin<<<1, GpuConfig::warpSize>>>(thrust::raw_pointer_cast(d_v.data()));

    h_v = d_v;
    thrust::host_vector<int> reference(GpuConfig::warpSize, 0);

    EXPECT_EQ(h_v, reference);
}

__global__ void testMax(int* values)
{
    int laneValue = threadIdx.x;

    values[threadIdx.x] = warpMax(laneValue);
}

TEST(WarpScan, max)
{
    thrust::host_vector<int> h_v(GpuConfig::warpSize);
    thrust::device_vector<int> d_v = h_v;

    testMax<<<1, GpuConfig::warpSize>>>(thrust::raw_pointer_cast(d_v.data()));

    h_v = d_v;
    thrust::host_vector<int> reference(GpuConfig::warpSize, GpuConfig::warpSize - 1);

    EXPECT_EQ(h_v, reference);
}

__global__ void testScan(int* values)
{
    int val             = 1;
    int scan            = inclusiveScanInt(val);
    values[threadIdx.x] = scan;
}

TEST(WarpScan, inclusiveInt)
{
    thrust::device_vector<int> d_values(2 * GpuConfig::warpSize);
    testScan<<<1, 2 * GpuConfig::warpSize>>>(rawPtr(d_values));
    thrust::host_vector<int> h_values = d_values;

    for (int i = 0; i < 2 * GpuConfig::warpSize; ++i)
    {
        EXPECT_EQ(h_values[i], i % GpuConfig::warpSize + 1);
    }
}

__global__ void testScanBool(int* result)
{
    bool val            = threadIdx.x % 2;
    result[threadIdx.x] = exclusiveScanBool(val);
}

TEST(WarpScan, bools)
{
    thrust::device_vector<int> d_values(2 * GpuConfig::warpSize);
    testScanBool<<<1, 2 * GpuConfig::warpSize>>>(rawPtr(d_values));
    thrust::host_vector<int> h_values = d_values;

    for (int i = 0; i < 2 * GpuConfig::warpSize; ++i)
    {
        EXPECT_EQ(h_values[i], (i % GpuConfig::warpSize) / 2);
    }
}

__global__ void testSegScan(int* values)
{
    int val = 1;

    if (threadIdx.x == 8) val = 2;

    if (threadIdx.x == 16) val = -2;

    if (threadIdx.x == 31) val = -3;

    int carry           = 1;
    int scan            = inclusiveSegscanInt(val, carry);
    values[threadIdx.x] = scan;
}

TEST(WarpScan, inclusiveSegInt)
{
    thrust::device_vector<int> d_values(GpuConfig::warpSize);
    testSegScan<<<1, GpuConfig::warpSize>>>(rawPtr(d_values));
    thrust::host_vector<int> h_values = d_values;

    //                         carry is one, first segment starts with offset of 1
    //                         |                                           | value(16) = -2, scan restarts at 2 - 1
    std::vector<int> reference{2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18,
                               1, 2, 3, 4, 5, 6, 7, 8, 9,  10, 11, 12, 13, 14, 15, 2};
    //                                                              value(31) = -3, scan restarts at 3 - 1  ^

    // we only check the first 32
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(h_values[i], reference[i]);
    }
}

__global__ void streamCompactTest(int* result)
{
    __shared__ int exchange[GpuConfig::warpSize];

    int val     = threadIdx.x;
    bool keep   = threadIdx.x % 2 == 0;
    int numKeep = streamCompact(&val, keep, exchange);

    result[threadIdx.x] = val;
}

TEST(WarpScan, streamCompact)
{
    thrust::device_vector<int> d_values(GpuConfig::warpSize);
    streamCompactTest<<<1, GpuConfig::warpSize>>>(rawPtr(d_values));
    thrust::host_vector<int> h_values = d_values;

    for (int i = 0; i < GpuConfig::warpSize / 2; ++i)
    {
        EXPECT_EQ(h_values[i], 2 * i);
    }
}

__global__ void spread(int* result)
{
    int val = 0;
    if (threadIdx.x < 4) val = result[threadIdx.x];

    result[threadIdx.x] = spreadSeg8(val);
}

TEST(WarpScan, spreadSeg8)
{
    thrust::device_vector<int> d_values(GpuConfig::warpSize);

    d_values[0] = 10;
    d_values[1] = 20;
    d_values[2] = 30;
    d_values[3] = 40;

    spread<<<1, GpuConfig::warpSize>>>(rawPtr(d_values));
    thrust::host_vector<int> h_values = d_values;

    thrust::host_vector<int> reference =
        std::vector<int>{10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27,
                         30, 31, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45, 46, 47};

    if (GpuConfig::warpSize == 64) // NOLINT
    {
        std::vector<int> tail{0, 1, 2, 3, 4, 5, 6, 7};
        for (int i = 0; i < 4; ++i)
        {
            std::copy(tail.begin(), tail.end(), std::back_inserter(reference));
        }
    }

    EXPECT_EQ(reference, h_values);
}
