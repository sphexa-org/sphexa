/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Tests for tuple gettres
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <array>

#include "gtest/gtest.h"

#include "cstone/fields/field_get.hpp"

using namespace cstone;

TEST(FieldGet, getPointers)
{
    std::vector<double> vd{1.0, 2.0, 3.0};
    std::vector<int> vi{1, 2, 3};

    int idx = 1;
    auto e1 = getPointers(std::tie(vd, vi), idx);

    *std::get<0>(e1) *= 2;
    *std::get<1>(e1) *= 3;

    EXPECT_EQ(vd[idx], 4.0);
    EXPECT_EQ(vi[idx], 6);
}
