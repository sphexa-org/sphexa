/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Utility tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/domain/index_ranges.hpp"

using namespace cstone;

TEST(IndexRanges, empty)
{
    IndexRanges<int> ranges;
    EXPECT_EQ(ranges.nRanges(), 0);
    EXPECT_EQ(ranges.totalCount(), 0);
}

TEST(IndexRanges, addRange)
{
    IndexRanges<int> ranges;
    ranges.addRange(0, 5);

    EXPECT_EQ(ranges.nRanges(), 1);
    EXPECT_EQ(ranges.rangeStart(0), 0);
    EXPECT_EQ(ranges.rangeEnd(0), 5);
    EXPECT_EQ(ranges.count(0), 5);

    ranges.addRange(10, 19);

    EXPECT_EQ(ranges.nRanges(), 2);
    EXPECT_EQ(ranges.totalCount(), 14);

    EXPECT_EQ(ranges.rangeStart(1), 10);
    EXPECT_EQ(ranges.rangeEnd(1), 19);
    EXPECT_EQ(ranges.count(1), 9);
}
