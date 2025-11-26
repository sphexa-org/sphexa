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

#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/device_vector.h"
#include "cstone/util/reallocate.hpp"

using namespace cstone;

TEST(DeviceVector, Construct)
{
    DeviceVector<int> v;

    std::size_t initSize = 10;
    v.resize(initSize);

    EXPECT_EQ(v.size(), initSize);
    EXPECT_EQ(v.capacity(), initSize);

    thrust::sequence(thrust::device, v.data(), v.data() + v.size(), 0);

    double growthFactor = 1.5;
    reallocate(v, 2 * initSize, growthFactor);

    EXPECT_EQ(v.size(), 2 * initSize);
    EXPECT_EQ(v.capacity(), 2 * initSize * growthFactor);

    thrust::sequence(thrust::device, v.data() + initSize, v.data() + v.size(), initSize);

    std::vector<int> h_v = toHost(v);

    std::vector<int> ref(v.size());
    std::iota(ref.begin(), ref.end(), 0);

    EXPECT_EQ(ref, h_v);
}

TEST(DeviceVector, PtrConstruct)
{
    std::vector<int> h{0, 1, 2, 3};
    DeviceVector<int> a(h.data(), h.data() + h.size());

    EXPECT_EQ(a.size(), h.size());
    EXPECT_EQ(a.capacity(), h.size());

    std::vector<int> dl = toHost(a);
    EXPECT_EQ(dl, h);
}

TEST(DeviceVector, Swap)
{
    DeviceVector<int> a(10);
    DeviceVector<int> b(20);

    int* aData = a.data();
    int* bData = b.data();

    swap(a, b);

    EXPECT_EQ(aData, b.data());
    EXPECT_EQ(bData, a.data());
}

TEST(DeviceVector, Assign)
{
    DeviceVector<int> a(10);
    DeviceVector<int> b(20);

    a = b;
    EXPECT_EQ(a.size(), b.size());

    a = DeviceVector<int>{};
    EXPECT_EQ(a.capacity(), 0);
}

TEST(DeviceVector, Capacity)
{
    DeviceVector<int> a(10);
    EXPECT_EQ(a.capacity(), 10);
    reallocate(a, 0, 1.0);
    EXPECT_EQ(a.size(), 0);
    EXPECT_EQ(a.capacity(), 10);
}

TEST(DeviceVector, ShrinkToFit)
{
    DeviceVector<int> a(10);
    EXPECT_EQ(a.capacity(), 10);
    reallocate(a, 8, 1.0);
    a.shrink_to_fit();
    EXPECT_EQ(a.size(), 8);
    EXPECT_EQ(a.capacity(), 8);
}
