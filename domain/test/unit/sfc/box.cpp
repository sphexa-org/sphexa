/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Test box functionality
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"
#include "cstone/sfc/box.hpp"

using namespace cstone;

TEST(SfcBox, pbcAdjust)
{
    EXPECT_EQ(pbcAdjust<1024>(-1024), 0);
    EXPECT_EQ(pbcAdjust<1024>(-1), 1023);
    EXPECT_EQ(pbcAdjust<1024>(0), 0);
    EXPECT_EQ(pbcAdjust<1024>(1), 1);
    EXPECT_EQ(pbcAdjust<1024>(1023), 1023);
    EXPECT_EQ(pbcAdjust<1024>(1024), 0);
    EXPECT_EQ(pbcAdjust<1024>(1025), 1);
    EXPECT_EQ(pbcAdjust<1024>(2047), 1023);
}

TEST(SfcBox, pbcDistance)
{
    int R = 1024;
    EXPECT_EQ(pbcDistance(-1024, R), 0);
    EXPECT_EQ(pbcDistance(-513, R), 511);
    EXPECT_EQ(pbcDistance(-512, R), 512);
    EXPECT_EQ(pbcDistance(-1, R), -1);
    EXPECT_EQ(pbcDistance(0, R), 0);
    EXPECT_EQ(pbcDistance(1, R), 1);
    EXPECT_EQ(pbcDistance(512, R), 512);
    EXPECT_EQ(pbcDistance(513, R), -511);
    EXPECT_EQ(pbcDistance(1024, R), 0);
}

TEST(SfcBox, applyPbc)
{
    using T = double;

    Box<T> box(0, 1, BoundaryType::periodic);
    Vec3<T> X{0.9, 0.9, 0.9};
    auto Xpbc = cstone::applyPbc(X, box);

    EXPECT_NEAR(Xpbc[0], -0.1, 1e-10);
    EXPECT_NEAR(Xpbc[1], -0.1, 1e-10);
    EXPECT_NEAR(Xpbc[2], -0.1, 1e-10);
}

TEST(SfcBox, putInBox)
{
    using T = double;
    {
        Box<T> box(0, 1, BoundaryType::periodic);
        Vec3<T> X{0.9, 0.9, 0.9};
        auto Xpbc = cstone::putInBox(X, box);

        EXPECT_NEAR(Xpbc[0], 0.9, 1e-10);
        EXPECT_NEAR(Xpbc[1], 0.9, 1e-10);
        EXPECT_NEAR(Xpbc[2], 0.9, 1e-10);
    }
    {
        Box<T> box(0, 1, BoundaryType::periodic);
        Vec3<T> X{1.1, 1.1, 1.1};
        auto Xpbc = cstone::putInBox(X, box);

        EXPECT_NEAR(Xpbc[0], 0.1, 1e-10);
        EXPECT_NEAR(Xpbc[1], 0.1, 1e-10);
        EXPECT_NEAR(Xpbc[2], 0.1, 1e-10);
    }
    {
        Box<T> box(-1, 1, BoundaryType::periodic);
        Vec3<T> X{-0.9, -0.9, -0.9};
        auto Xpbc = cstone::putInBox(X, box);

        EXPECT_NEAR(Xpbc[0], -0.9, 1e-10);
        EXPECT_NEAR(Xpbc[1], -0.9, 1e-10);
        EXPECT_NEAR(Xpbc[2], -0.9, 1e-10);
    }
}

TEST(SfcBox, createIBox)
{
    {
        using T                = double;
        using KeyType          = uint32_t;
        constexpr int maxCoord = 1u << maxTreeLevel<KeyType>{};

        Box<T> box(0, 1);

        T r = T(1.0) / maxCoord;
        T c = 1.0 - 0.5 * r;
        T s = 0.5 * r;
        Vec3<T> aCenter{c, c, c};
        Vec3<T> aSize{s, s, s};

        IBox probe = createIBox<KeyType>(aCenter, aSize, box);
        IBox ref{maxCoord - 1, maxCoord};
        EXPECT_EQ(ref, probe);
    }
    {
        using T       = double;
        using KeyType = uint64_t;

        Box<T> box(-1, 1, -2, 2, -3, 3);
        Vec3<T> aCenter{0.1, 0.2, 0.3};
        Vec3<T> aSize{0.01, 0.02, 0.03};

        IBox probe = createIBox<KeyType>(aCenter, aSize, box);
        IBox ref{1142947, 1163920};
        EXPECT_EQ(ref, probe);
    }
}

template<typename T>
static bool contains(const Box<T>& large_box, const Box<T>& small_box)
{
    return (large_box.xmin() <= small_box.xmin() && large_box.ymin() <= small_box.ymin() &&
            large_box.zmin() <= small_box.zmin() && large_box.xmax() >= small_box.xmax() &&
            large_box.ymax() >= small_box.ymax() && large_box.zmax() >= small_box.zmax());
};

//! @brief newer box bigger than old box, limitBoxShrink has no effect
TEST(limitBox, expand)
{
    using T                  = double;
    constexpr T shrinkFactor = 0.1;
    auto pbc                 = BoundaryType::periodic;
    auto open                = BoundaryType::open;
    Box<T> previousBox(0, 1, 2, 3, 4, 5, open, pbc, open);
    Box<T> currentBox(-1, 2, -2, 3, -4, 6, open, pbc, open);
    Box<T> limitedBox = limitBoxShrinking(currentBox, previousBox, shrinkFactor);
    EXPECT_EQ(limitedBox, currentBox);
}

//! @brief newer box bigger than shrink limit, limitBoxShrink has no effect
TEST(limitBox, aboveShrinkLimit)
{
    using T                  = double;
    constexpr T shrinkFactor = 0.1001;
    auto pbc                 = BoundaryType::periodic;
    auto open                = BoundaryType::open;
    Box<T> previousBox(0, 1, 2, 3, 4, 5, open, pbc, open);
    Box<T> currentBox(0.1, 0.9, 2.1, 2.9, 4.1, 4.9, open, pbc, open);
    Box<T> limitedBox = limitBoxShrinking(currentBox, previousBox, shrinkFactor);
    EXPECT_EQ(limitedBox, currentBox);
}

//! @brief newer box smaller than shrink limit, limitBoxShrink kicks in
TEST(limitBox, belowShrinkLimit)
{
    using T                  = double;
    constexpr T shrinkFactor = 0.05;
    Box<T> previousBox(1, 2, 10, 20, 100, 200);
    Box<T> currentBox(1.1, 1.9, 11, 19, 110, 190);
    Box<T> limitedBox = limitBoxShrinking(currentBox, previousBox, shrinkFactor);
    EXPECT_NEAR(limitedBox.xmin(), 1.05, 1e-6);
    EXPECT_NEAR(limitedBox.xmax(), 1.95, 1e-6);
    EXPECT_NEAR(limitedBox.ymin(), 10.5, 1e-6);
    EXPECT_NEAR(limitedBox.ymax(), 19.5, 1e-6);
    EXPECT_NEAR(limitedBox.zmin(), 105, 1e-6);
    EXPECT_NEAR(limitedBox.zmax(), 195, 1e-6);

    EXPECT_TRUE(contains(previousBox, limitedBox));
}

TEST(SfcBox, limitBoxShrinking)
{
    using T                   = double;
    constexpr T shrink_factor = 0.1;
    Box<T> previousBox(0., 2., -1., 1., -1., 5.);
    Box<T> currentBox(0., 0.05, 0., 1., 1., 2.);
    Box<T> limitedBox = limitBoxShrinking(currentBox, previousBox, shrink_factor);

    EXPECT_NEAR(limitedBox.lx(), (1. - shrink_factor) * previousBox.lx(), 1e-6);
    EXPECT_NEAR(limitedBox.ly(), (1. - shrink_factor) * previousBox.ly(), 1e-6);
    EXPECT_NEAR(limitedBox.lz(), (1. - 2. * shrink_factor) * previousBox.lz(), 1e-6);

    EXPECT_TRUE(contains(limitedBox, currentBox));
}
