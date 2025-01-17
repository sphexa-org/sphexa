/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurichc
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief math tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"
#include "cstone/primitives/math.hpp"

using namespace cstone;

TEST(Math, round_up)
{
    EXPECT_EQ(round_up(127, 128), 128);
    EXPECT_EQ(round_up(128, 128), 128);
    EXPECT_EQ(round_up(129, 128), 256);
    EXPECT_EQ(round_up(257, 128), 384);

    EXPECT_EQ(round_up(127lu, 128lu), 128lu);
    EXPECT_EQ(round_up(128lu, 128lu), 128lu);
    EXPECT_EQ(round_up(129lu, 128lu), 256lu);
}

TEST(Math, butterfly)
{
    std::array<uint32_t, 9> W1;
    for (uint32_t i = 0; i < W1.size(); ++i)
    {
        W1[i] = butterfly(i);
    }

    std::array<uint32_t, 9> ref{0, 1, 2, 1, 3, 1, 2, 1, 4};
    EXPECT_EQ(W1, ref);
}
