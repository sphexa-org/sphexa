/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Halo exchange auxiliary functions GPU testing
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#include "gtest/gtest.h"

#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cstone/cuda/thrust_util.cuh"
#include "cstone/halos/gather_halos_gpu.h"

using namespace cstone;

TEST(Halos, gatherRanges)
{
    // list of marked halo cells/ranges
    std::vector<int> seq(30);
    std::iota(seq.begin(), seq.end(), 0);
    thrust::device_vector<int> src = seq;

    thrust::device_vector<unsigned> rangeScan    = std::vector<unsigned>{0, 4, 7};
    thrust::device_vector<unsigned> rangeOffsets = std::vector<unsigned>{4, 12, 22};
    int totalCount                               = 10;

    thrust::device_vector<int> buffer = std::vector<int>(totalCount);

    gatherRanges(rawPtr(rangeScan), rawPtr(rangeOffsets), rangeScan.size(), rawPtr(src), rawPtr(buffer), totalCount);

    thrust::host_vector<int> h_buffer = buffer;
    thrust::host_vector<int> ref      = std::vector<int>{4, 5, 6, 7, 12, 13, 14, 22, 23, 24};

    EXPECT_EQ(h_buffer, ref);
}
