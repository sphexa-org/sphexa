/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Random coordinates generation for testing
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <gtest/gtest.h>

#include "random.hpp"

using namespace cstone;

TEST(CoordinateSamples, randomContainerIsSorted)
{
    using real        = double;
    using IntegerType = unsigned;
    int n             = 10;

    Box<real> box{0, 1, -1, 2, 0, 5};
    RandomCoordinates<real, SfcKind<IntegerType>> c(n, box);

    std::vector<IntegerType> testCodes(n);
    computeSfcKeys(c.x().data(), c.y().data(), c.z().data(), sfcKindPointer(testCodes.data()), n, box);

    EXPECT_TRUE(std::is_sorted(testCodes.begin(), testCodes.end()));
}
