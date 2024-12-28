/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Test continuum octree generation
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/tree/continuum.hpp"
#include "cstone/tree/cs_util.hpp"

using namespace cstone;

TEST(CsConcentration, concentrationCountConstant)
{
    using T       = double;
    using KeyType = uint32_t;

    Box<T> box(-1, 1);
    size_t N0 = 1000;

    auto constRho = [N0](T, T, T) { return T(N0) / 8.0; };

    size_t count = continuumCount(KeyType(0), nodeRange<KeyType>(0), box, constRho);

    EXPECT_EQ(count, N0);
}

TEST(CsConcentration, computeTreeOneOverR)
{
    using T       = double;
    using KeyType = uint64_t;

    unsigned bucketSize = 64;
    Box<T> box(-1, 1);
    T eps     = box.lx() / (1u << maxTreeLevel<KeyType>{});
    size_t N0 = 1000000;

    auto oneOverR = [N0, eps](T x, T y, T z)
    {
        T r = std::max(std::sqrt(norm2(Vec3<T>{x, y, z})), eps);
        if (r > 1.0) { return 0.0; }
        else { return T(N0) / (2 * M_PI * r); }
    };

    auto [tree, counts] = computeContinuumCsarray<KeyType>(oneOverR, box, bucketSize);
    size_t totalCount   = std::accumulate(counts.begin(), counts.end(), 0lu);

    for (auto c : counts)
    {
        EXPECT_LT(c, 1.5 * bucketSize);
    }

    EXPECT_NEAR(totalCount, N0, N0 * 0.03);
}
