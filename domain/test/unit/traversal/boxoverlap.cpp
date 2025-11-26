/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Box overlap tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/traversal/boxoverlap.hpp"

using namespace cstone;

TEST(BoxOverlap, overlapRange)
{
    constexpr int R = 1024;

    EXPECT_TRUE(overlapRange(0, 2, 1, 3, R));
    EXPECT_FALSE(overlapRange(0, 1, 1, 2, R));
    EXPECT_FALSE(overlapRange(0, 1, 2, 3, R));
    EXPECT_TRUE(overlapRange(0, 1023, 1, 3, R));
    EXPECT_TRUE(overlapRange(0, 1024, 1, 3, R));
    EXPECT_TRUE(overlapRange(0, 2048, 1, 3, R));

    EXPECT_TRUE(overlapRange(1022, 1024, 1023, 1024, R));
    EXPECT_TRUE(overlapRange(1023, 1025, 0, 1, R));
    EXPECT_FALSE(overlapRange(0, 1, 1023, 1024, R));
    EXPECT_TRUE(overlapRange(-1, 1, 1023, 1024, R));
    EXPECT_FALSE(overlapRange(-1, 1, 1022, 1023, R));

    EXPECT_TRUE(overlapRange(1023, 2048, 0, 1, R));
    EXPECT_TRUE(overlapRange(512, 1024, 332, 820, R));
}

TEST(BoxOverlap, rangeSep)
{
    EXPECT_EQ(rangeSep(0, 1, 3, 4, 10), 2);
    EXPECT_EQ(rangeSep(0, 1, 3, 4, 5), 1);
    EXPECT_EQ(rangeSep(0, 1, 3, 4, 0), 2);

    EXPECT_EQ(rangeSep(3, 4, 0, 1, 10), 2);
    EXPECT_EQ(rangeSep(3, 4, 0, 1, 5), 1);
    EXPECT_EQ(rangeSep(3, 4, 0, 1, 0), 2);
}

TEST(BoxOverlap, boxSeparation)
{
    {
        Vec3<int> pbc{1024, 1024, 1024};
        Vec3<int> ref{2, 2, 2};
        EXPECT_EQ(boxSeparation({0, 1}, {3, 6}, pbc), ref);
    }
    {
        Vec3<int> pbc{0, 1024, 1024};
        Vec3<int> ref{2, 1, 1};
        EXPECT_EQ(boxSeparation({0, 1}, {3, 1023}, pbc), ref);
    }
    {
        Vec3<int> pbc{0, 100, 200};
        Vec3<int> ref{2, 4, 7};
        EXPECT_EQ(boxSeparation({0, 1, 10, 100, 2, 4}, {3, 5, 4, 5, 180, 195}, pbc), ref);
    }
    {
        Vec3<int> pbc{1024, 1024, 1024};
        Vec3<int> ref{0, 0, 0};
        EXPECT_EQ(boxSeparation({0, 2, 2, 4, 6, 8}, {1, 3, 3, 5, 7, 9}, pbc), ref);
    }
}

/*! @brief Test overlap between octree nodes and coordinate ranges
 *
 * The octree node is given as a Morton code plus number of bits
 * and the coordinates as integer ranges.
 */
template<class KeyType>
void overlapTest()
{
    unsigned level = 2;
    // range of a level-2 node
    int r = KeyType(1) << (maxTreeLevel<KeyType>{} - level);

    // node range: [r,2r]^3
    IBox target(r, 2 * r, r, 2 * r, r, 2 * r);

    /// Each test is a separate case

    EXPECT_FALSE(overlap<KeyType>(target, IBox{0, r, 0, r, 0, r}));

    // exact match
    EXPECT_TRUE(overlap<KeyType>(target, IBox{r, 2 * r, r, 2 * r, r, 2 * r}));
    // contained within (1,1,1) corner of node
    EXPECT_TRUE(overlap<KeyType>(target, IBox{2 * r - 1, 2 * r, 2 * r - 1, 2 * r, 2 * r - 1, 2 * r}));
    // contained and exceeding (1,1,1) corner by 1 in all dimensions
    EXPECT_TRUE(overlap<KeyType>(target, IBox{2 * r - 1, 2 * r + 1, 2 * r - 1, 2 * r + 1, 2 * r - 1, 2 * r + 1}));

    // all of these miss the (1,1,1) corner by 1 in one of the three dimensions
    EXPECT_FALSE(overlap<KeyType>(target, IBox{2 * r, 2 * r + 1, 2 * r - 1, 2 * r, 2 * r - 1, 2 * r}));
    EXPECT_FALSE(overlap<KeyType>(target, IBox{2 * r - 1, 2 * r, 2 * r, 2 * r + 1, 2 * r - 1, 2 * r}));
    EXPECT_FALSE(overlap<KeyType>(target, IBox{2 * r - 1, 2 * r, 2 * r - 1, 2 * r, 2 * r, 2 * r + 1}));

    // contained within (0,0,0) corner of node
    EXPECT_TRUE(overlap<KeyType>(target, IBox{r, r + 1, r, r + 1, r, r + 1}));

    // all of these miss the (0,0,0) corner by 1 in one of the three dimensions
    EXPECT_FALSE(overlap<KeyType>(target, IBox{r - 1, r, r, r + 1, r, r + 1}));
    EXPECT_FALSE(overlap<KeyType>(target, IBox{r, r + 1, r - 1, r, r, r + 1}));
    EXPECT_FALSE(overlap<KeyType>(target, IBox{r, r + 1, r, r + 1, r - 1, r}));
}

TEST(BoxOverlap, overlaps)
{
    overlapTest<unsigned>();
    overlapTest<uint64_t>();
}

//! @brief test overlaps of periodic halo boxes with parts of the SFC tree
template<class KeyType>
void pbcOverlaps()
{
    int maxCoord = 1u << maxTreeLevel<KeyType>{};
    {
        IBox boxA{-1, 1, 0, 1, 0, 1};
        IBox boxB{0, 1, 0, 1, 0, 1};
        EXPECT_TRUE(overlap<KeyType>(boxA, boxB));
    }
    {
        IBox haloBox{-1, 1, 0, 1, 0, 1};
        IBox corner{maxCoord - 1, maxCoord, 0, 1, 0, 1};
        EXPECT_TRUE(overlap<KeyType>(corner, haloBox));
    }
    {
        IBox haloBox{maxCoord - 1, maxCoord + 2, 0, 1, 0, 1};
        IBox corner{0, 1, 0, 1, 0, 1};
        EXPECT_TRUE(overlap<KeyType>(corner, haloBox));
    }
    {
        IBox haloBox{-1, 1, -1, 1, -1, 1};
        IBox corner{maxCoord - 1, maxCoord};
        EXPECT_TRUE(overlap<KeyType>(corner, haloBox));
    }
}

TEST(BoxOverlap, pbcOverlaps)
{
    pbcOverlaps<unsigned>();
    pbcOverlaps<uint64_t>();
}

//! @brief check halo box ranges in all spatial dimensions
template<class KeyType>
void makeHaloBoxXYZ()
{
    constexpr int maxCoord = 1 << maxTreeLevel<KeyType>{};
    int r                  = KeyType(1) << (maxTreeLevel<KeyType>{} - 3);

    Box<float> box(0, 1, 0, 0.5, 0, 1.0 / 3);
    IBox nodeBox(r, 2 * r, r, 2 * r, r, 2 * r);

    IBox haloBox = makeHaloBox<KeyType>(nodeBox, 1.0 / maxCoord, box);
    IBox refBox{r - 1, 2 * r + 1, r - 2, 2 * r + 2, r - 3, 2 * r + 3};
    EXPECT_EQ(haloBox, refBox);
}

TEST(BoxOverlap, makeHaloBoxXYZ)
{
    makeHaloBoxXYZ<unsigned>();
    makeHaloBoxXYZ<uint64_t>();
}

//! @brief underflow check, non-periodic case
template<class KeyType>
void makeHaloBoxUnderflow()
{
    constexpr int maxCoord = 1 << maxTreeLevel<KeyType>{};
    int r                  = KeyType(1) << (maxTreeLevel<KeyType>{} - 1);

    Box<float> box(0, 1);
    IBox nodeBox(0, r);

    IBox haloBox = makeHaloBox<KeyType>(nodeBox, 0.99 / maxCoord, box);
    IBox refBox{0, r + 1, 0, r + 1, 0, r + 1};
    EXPECT_EQ(haloBox, refBox);
}

TEST(BoxOverlap, makeHaloBoxUnderflow)
{
    makeHaloBoxUnderflow<unsigned>();
    makeHaloBoxUnderflow<uint64_t>();
}

//! @brief overflow check, non-periodic case
template<class KeyType>
void makeHaloBoxOverflow()
{
    constexpr int maxCoord = 1 << maxTreeLevel<KeyType>{};
    int r                  = KeyType(1) << (maxTreeLevel<KeyType>{} - 1);

    IBox nodeBox(r, 2 * r, r, 2 * r, r, 2 * r);
    Box<float> box(0, 1);

    IBox haloBox = makeHaloBox<KeyType>(nodeBox, 0.99 / maxCoord, box);
    IBox refBox{r - 1, 2 * r, r - 1, 2 * r, r - 1, 2 * r};
    EXPECT_EQ(haloBox, refBox);
}

TEST(BoxOverlap, makeHaloBoxOverflow)
{
    makeHaloBoxOverflow<unsigned>();
    makeHaloBoxOverflow<uint64_t>();
}

//! @brief check halo box ranges with periodic boundary conditions
template<class KeyType>
void makeHaloBoxPbc()
{
    int r = 1 << (maxTreeLevel<KeyType>{} - 3);

    IBox nodeBox(r, 2 * r, r, 2 * r, r, 2 * r);
    Box<double> bbox(0, 1, cstone::BoundaryType::periodic);

    {
        double radius = 0.999 / r; // normalize(radius) = 7.992
        IBox haloBox  = makeHaloBox<KeyType>(nodeBox, radius, bbox);
        IBox refBox{r - 8, 2 * r + 8, r - 8, 2 * r + 8, r - 8, 2 * r + 8};
        EXPECT_EQ(haloBox, refBox);
    }
    {
        double radius = 1.000001 / 8; // normalize(radius) = r + epsilon
        IBox haloBox  = makeHaloBox<KeyType>(nodeBox, radius, bbox);
        IBox refBox{-1, 3 * r + 1, -1, 3 * r + 1, -1, 3 * r + 1};
        EXPECT_EQ(haloBox, refBox);
    }
}

TEST(BoxOverlap, makeHaloBoxPbc)
{
    makeHaloBoxPbc<unsigned>();
    makeHaloBoxPbc<uint64_t>();
}

template<class I>
void haloBoxContainedIn()
{
    {
        IBox haloBox{0, 1, 0, 1, 0, 1};
        EXPECT_TRUE(containedIn(I(0), I(1), haloBox));
    }
    {
        IBox haloBox{0, 1, 0, 1, 0, 2};
        EXPECT_FALSE(containedIn(I(0), I(1), haloBox));
    }
    {
        IBox haloBox{0, 1, 0, 1, 0, 2};
        EXPECT_TRUE(containedIn(I(0), I(8), haloBox));
    }
    {
        IBox haloBox{0, 1, 0, 2, 0, 2};
        EXPECT_FALSE(containedIn(I(0), I(3), haloBox));
    }
    {
        IBox haloBox{0, 1, 0, 2, 0, 2};
        EXPECT_TRUE(containedIn(I(0), I(8), haloBox));
    }
    {
        IBox haloBox{0, 2, 0, 2, 0, 2};
        EXPECT_FALSE(containedIn(I(0), I(7), haloBox));
    }
    {
        IBox haloBox{0, 2, 0, 2, 0, 2};
        EXPECT_TRUE(containedIn(I(0), I(8), haloBox));
    }

    /// PBC
    {
        IBox haloBox{-1, 1, 0, 1, 0, 1};
        EXPECT_FALSE(containedIn(I(0), I(1), haloBox));
    }
}

//! @brief test containment of a box within a Morton code range
TEST(BoxOverlap, haloBoxContainedIn)
{
    haloBoxContainedIn<unsigned>();
    haloBoxContainedIn<uint64_t>();
}

template<class KeyType>
void excludeRangeContainedIn()
{
    KeyType rangeStart = pad(KeyType(01), 3);
    KeyType rangeEnd   = pad(KeyType(02), 3);

    {
        KeyType prefix = 0b1001;
        EXPECT_TRUE(containedIn(prefix, rangeStart, rangeEnd));
    }
    {
        KeyType prefix = 0b10010;
        EXPECT_TRUE(containedIn(prefix, rangeStart, rangeEnd));
    }
    {
        KeyType prefix = 0b1000;
        EXPECT_FALSE(containedIn(prefix, rangeStart, rangeEnd));
    }
    {
        KeyType prefix = 1;
        EXPECT_FALSE(containedIn(prefix, rangeStart, rangeEnd));
    }

    rangeStart = 0;
    rangeEnd   = pad(KeyType(01), 3);
    {
        KeyType prefix = 0b1000;
        EXPECT_TRUE(containedIn(prefix, rangeStart, rangeEnd));
    }
    {
        KeyType prefix = 0b100;
        EXPECT_FALSE(containedIn(prefix, rangeStart, rangeEnd));
    }
}

TEST(BoxOverlap, excludeRangeContainedIn)
{
    excludeRangeContainedIn<unsigned>();
    excludeRangeContainedIn<uint64_t>();
}

TEST(BoxOverlap, containedInFP)
{
    using T       = double;
    using KeyType = uint64_t;

    Box<T> box(0, 1);

    Vec3<T> center{0.375, 0.375, 0.375}, size{0.335, 0.335, 0.335};

    KeyType exclStart = 0;
    KeyType exclEnd   = 060000000000000000000;

    EXPECT_FALSE(containedIn(exclStart, exclEnd, center, size, box));
}

TEST(BoxOverlap, insideBox)
{
    using T = double;
    Box<T> box(0, 1);
    {
        Vec3<T> bcenter{0.75, 0.25, 0.25};
        Vec3<T> bsize{0.25, 0.25, 0.25};
        EXPECT_TRUE(insideBox(bcenter, bsize, box));
    }
    {
        Vec3<T> bcenter{0.75, 0.25, 0.25};
        Vec3<T> bsize{0.26, 0.25, 0.25};
        EXPECT_FALSE(insideBox(bcenter, bsize, box));
    }
    {
        Vec3<T> bcenter{0.1, 0.1, 0.1};
        Vec3<T> bsize{0.1, 0.11, 0.1};
        EXPECT_FALSE(insideBox(bcenter, bsize, box));
    }
}

TEST(BoxOverlap, minPointDistance)
{
    using T       = double;
    using KeyType = unsigned;

    constexpr unsigned mc = maxCoord<KeyType>{};

    {
        Box<T> box(0, 1);
        IBox ibox(0, mc / 2);

        T px = (mc / 2.0 + 1) / mc;
        Vec3<T> X{px, px, px};

        auto [center, size] = centerAndSize<KeyType>(ibox, box);

        T probe = std::sqrt(norm2(minDistance(X, center, size, box)));
        EXPECT_NEAR(std::sqrt(3) / mc, probe, 1e-10);
    }
}

TEST(BoxOverlap, minDistance)
{
    using T = double;

    {
        Box<T> box(0, 2, 0, 3, 0, 4);

        Vec3<T> aCenter{1., 1., 1.};
        Vec3<T> bCenter{1., 2., 3.};

        Vec3<T> aSize{0.1, 0.1, 0.1};
        Vec3<T> bSize{0.1, 0.1, 0.1};

        Vec3<T> dist = minDistance(aCenter, aSize, bCenter, bSize, box);
        EXPECT_NEAR(dist[0], 0., 1e-10);
        EXPECT_NEAR(dist[1], 0.8, 1e-10);
        EXPECT_NEAR(dist[2], 1.8, 1e-10);
    }
    {
        Box<T> boxPbc(0, 2, 0, 3, 0, 4, BoundaryType::periodic, BoundaryType::periodic, BoundaryType::periodic);

        Vec3<T> aCenter{0.1, 0.1, 0.1};
        Vec3<T> bCenter{1.9, 2.9, 3.9};

        Vec3<T> aSize{0.1, 0.1, 0.1};
        Vec3<T> bSize{0.1, 0.1, 0.1};

        Vec3<T> dist = minDistance(aCenter, aSize, bCenter, bSize, boxPbc);
        EXPECT_NEAR(dist[0], 0., 1e-10);
        EXPECT_NEAR(dist[1], 0., 1e-10);
        EXPECT_NEAR(dist[2], 0., 1e-10);
    }
}
