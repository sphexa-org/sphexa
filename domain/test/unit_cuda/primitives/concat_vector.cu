/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/device_vector.h"
#include "cstone/primitives/concat_vector.hpp"

using namespace cstone;

TEST(PrimitivesGpu, concatVector)
{
    ConcatVector<int> v;
    v.reindex({1, 2, 3});

    auto modView = v.view();
    std::iota(modView[0].begin(), modView[0].end(), 10);
    std::iota(modView[1].begin(), modView[1].end(), 20);
    std::iota(modView[2].begin(), modView[2].end(), 30);

    ConcatVector<int, DeviceVector> d_v;
    copy(v, d_v);

    ConcatVector<int> probe;
    copy(d_v, probe);

    auto probeView = probe.view();
    EXPECT_EQ(probeView[2][0], 30);
    EXPECT_EQ(probe, v);
}
