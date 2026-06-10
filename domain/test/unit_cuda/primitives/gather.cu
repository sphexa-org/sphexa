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

#include <cstone/cuda/device_vector.h>
#include "cstone/primitives/primitives_acc.hpp"

using namespace cstone;

TEST(SortByKey, minimal)
{
    using KeyType   = unsigned;
    using IndexType = unsigned;

    DeviceVector<KeyType> keys = std::vector<KeyType>{2, 1, 5, 4};
    DeviceVector<IndexType> obuf, keyBuf, valBuf;

    constexpr execution::Gpu stream{0};

    LocalIndex off = 1;
    sequence(off, keys.size(), obuf, 1.0, stream);
    sortByKey(std::span{keys.data(), keys.size()}, std::span{obuf.data() + off, keys.size()}, keyBuf, valBuf, 1.0, stream);
    // map is [. 2 1 4 3]

    sequence(0, off, obuf, 1.0, stream);
    {
        DeviceVector ref = std::vector<IndexType>{0, 2, 1, 4, 3};
        EXPECT_EQ(obuf, ref);
    }
}
