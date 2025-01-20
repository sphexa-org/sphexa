/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
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
