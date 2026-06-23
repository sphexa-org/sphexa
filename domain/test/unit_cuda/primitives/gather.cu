/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief GPU SFC sorter unit tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#include <vector>

#include "gtest/gtest.h"

#include "cstone/cuda/device_vector.h"
#include "cstone/cuda/errorcheck.cuh"
#include "cstone/cuda/stream_holder.cuh"
#include "cstone/primitives/primitives_acc.hpp"

using namespace cstone;

TEST(SortByKey, minimal)
{
    using KeyType   = unsigned;
    using IndexType = unsigned;

    DeviceVector<KeyType> keys = std::vector<KeyType>{2, 1, 5, 4};
    DeviceVector<IndexType> obuf, keyBuf, valBuf;

    StreamHolder stream;

    LocalIndex off = 1;
    sequence(stream.exec(), off, keys.size(), obuf, 1.0);
    sortByKey(stream.exec(), std::span{keys.data(), keys.size()}, std::span{obuf.data() + off, keys.size()}, keyBuf,
              valBuf, 1.0);
    // map is [. 2 1 4 3]

    sequence(stream.exec(), 0, off, obuf, 1.0);
    {
        DeviceVector ref = std::vector<IndexType>{0, 2, 1, 4, 3};
        EXPECT_EQ(obuf, ref);
    }
}
