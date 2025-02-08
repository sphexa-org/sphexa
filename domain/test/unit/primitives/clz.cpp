/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Count leading zero tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>
#include "gtest/gtest.h"

#include "cstone/primitives/clz.hpp"

TEST(CLZ, C_clz_32)
{
    std::vector<unsigned> inputs{0xF0000000, 0x70000000, 1};
    std::vector<unsigned> references{0, 1, 31};

    std::vector<unsigned> probes;
    for (unsigned i : inputs)
        probes.push_back(detail::clz32(i));

    EXPECT_EQ(probes, references);
}

TEST(CLZ, C_clz_64)
{
    std::vector<uint64_t> inputs{1ul << 63u, (1ul << 62u) + 722302, 5, 4, 2, 1};
    std::vector<int> references{0, 1, 61, 61, 62, 63};

    std::vector<int> probes;
    for (auto i : inputs)
        probes.push_back(detail::clz64(i));

    EXPECT_EQ(probes, references);
}

TEST(CLZ, builtin_clz_32)
{
    std::vector<unsigned> inputs{0xF0000000, 0x70000000, 1, 0};
    std::vector<unsigned> references{0, 1, 31, 32};

    std::vector<unsigned> probes;
    for (unsigned i : inputs)
        probes.push_back(countLeadingZeros(i));

    EXPECT_EQ(probes, references);
}

TEST(CLZ, builtin_clz_64)
{
    std::vector<uint64_t> inputs{1ul << 63u, (1ul << 62u) + 23427, 5, 4, 2, 1, 0};
    std::vector<int> references{0, 1, 61, 61, 62, 63, 64};

    std::vector<int> probes;
    for (auto i : inputs)
        probes.push_back(countLeadingZeros(i));

    EXPECT_EQ(probes, references);
}
