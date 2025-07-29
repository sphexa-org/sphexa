/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Compare and test different multipole approximations
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/focus/source_center.hpp"

#include "dataset.hpp"
#include "ryoanji/nbody/cartesian_qpole.hpp"
#include "ryoanji/nbody/kernel.hpp"

using namespace ryoanji;

//! @brief Tests direct particle-to-particle gravity interactions with mass softening
TEST(Gravity, P2PmsoftBase)
{
    using T = double;

    cstone::Vec3<T> target{1, 1, 1};
    T               h = std::sqrt(3) / 2 - 0.001;

    cstone::Vec3<T> source1{2, 2, 2}, source2{-2, -2, -2};

    cstone::Vec4<T> acc{0, 0, 0, 0};
    acc = P2P(acc, target, source1, 1.0, h, h);
    acc = P2P(acc, target, source2, 1.0, h, h);

    // h too small to trigger softening, so results should match the non-softened numbers
    EXPECT_NEAR(acc[0], -0.76980035891950138, 1e-10);
    EXPECT_NEAR(acc[1], 0.17106674642655587, 1e-10);
    EXPECT_NEAR(acc[2], 0.17106674642655587, 1e-10);
    EXPECT_NEAR(acc[3], 0.17106674642655587, 1e-10);
}

//! @brief Tests direct particle-to-particle gravity interactions with mass softening
TEST(Gravity, P2PmsoftH)
{
    using T = double;

    cstone::Vec3<T> target{1, 1, 1};
    T               h = std::sqrt(3) / 2 + 0.001;

    cstone::Vec3<T> source1{2, 2, 2}, source2{-2, -2, -2};

    cstone::Vec4<T> acc{0, 0, 0, 0};
    acc = P2P(acc, target, source1, 1.0, h, h);
    acc = P2P(acc, target, source2, 1.0, h, h);

    EXPECT_NEAR(acc[0], -0.7678049688481372, 1e-10);
    EXPECT_NEAR(acc[1], 0.1704016164027678, 1e-10);
    EXPECT_NEAR(acc[2], 0.1704016164027678, 1e-10);
    EXPECT_NEAR(acc[3], 0.1704016164027678, 1e-10);
}

//! @brief The traversal code relies on P2P self interaction being zero
TEST(Gravity, P2PselfInteraction)
{
    using T = double;

    Vec3<T> pos_i{0, 0, 0};
    T       h = 0.1;
    Vec4<T> acc{0, 0, 0, 0};

    auto self = P2P(acc, pos_i, pos_i, 1.0, h, h);
    EXPECT_EQ(self, acc);
}
