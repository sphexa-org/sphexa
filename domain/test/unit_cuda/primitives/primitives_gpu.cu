/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Cornerstone octree GPU testing
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#include <numeric>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "gtest/gtest.h"

#include "cstone/cuda/thrust_util.cuh"
#include "cstone/primitives/primitives_gpu.h"

using namespace cstone;

TEST(PrimitivesGpu, MinMax)
{
    using thrust::raw_pointer_cast;

    thrust::device_vector<double> v = std::vector<double>{1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};

    auto minMax = MinMaxGpu<double>{}(raw_pointer_cast(v.data()), raw_pointer_cast(v.data()) + v.size());

    EXPECT_EQ(std::get<0>(minMax), 1.);
    EXPECT_EQ(std::get<1>(minMax), 10.);
}
