/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Test hilbert code implementation
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <algorithm>
#include <array>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "cstone/sfc/hilbert.hpp"
#include "cstone/sfc/sfc.hpp"

using namespace cstone;

//! @brief tests numKeys random 3D points for encoding/decoding consistency
template<class KeyType>
void inversionTest2D()
{
    int numKeys  = 1000;
    int maxCoord = (1 << maxTreeLevel<KeyType>{}) - 1;

    std::mt19937 gen;
    std::uniform_int_distribution<unsigned> distribution(0, maxCoord);

    auto getRand = [&distribution, &gen]() { return distribution(gen); };

    std::vector<unsigned> x(numKeys);
    std::vector<unsigned> y(numKeys);

    std::generate(begin(x), end(x), getRand);
    std::generate(begin(y), end(y), getRand);

    for (int i = 0; i < numKeys; ++i)
    {
        KeyType hilbertKey = iHilbert2D<KeyType>(x[i], y[i]);
        auto [a, b]        = decodeHilbert2D<KeyType>(hilbertKey);
        EXPECT_EQ(x[i], a);
        EXPECT_EQ(y[i], b);
    }
}

TEST(HilbertCode, inversion2D)
{
    inversionTest2D<unsigned>();
    inversionTest2D<uint64_t>();
}

//! @brief 2D test the 2D Hilbert curve of order 1 and 2
template<class KeyType>
void Hilbert2D(int order)
{
    for (unsigned xi = 0; xi < pow(2, order); ++xi)
    {
        for (unsigned yi = 0; yi < pow(2, order); ++yi)
        {

            KeyType key = iHilbert2D<KeyType>(xi, yi);
            auto t      = decodeHilbert2D<KeyType>(key);
            EXPECT_EQ(xi, util::get<0>(t));
            EXPECT_EQ(yi, util::get<1>(t));
        }
    }
}

TEST(HilbertCode, Hilbert2D)
{
    Hilbert2D<unsigned>(2);
    Hilbert2D<uint64_t>(1);
}

//! @brief test the curve on the first eight octants
template<class KeyType>
void firstOrderCurve()
{
    constexpr unsigned hilbertToMorton[8] = {0, 1, 3, 2, 6, 7, 5, 4};

    for (unsigned xi = 0; xi < 2; ++xi)
    {
        for (unsigned yi = 0; yi < 2; ++yi)
        {
            for (unsigned zi = 0; zi < 2; ++zi)
            {
                unsigned L1Range      = (1 << maxTreeLevel<KeyType>{}) / 2;
                unsigned mortonOctant = 4 * xi + 2 * yi + zi;

                {
                    KeyType hilbertKey     = iHilbert<KeyType>(L1Range * xi, L1Range * yi, L1Range * zi);
                    unsigned hilbertOctant = octalDigit(hilbertKey, 1);
                    EXPECT_EQ(mortonOctant, hilbertToMorton[hilbertOctant]);
                }
                {
                    KeyType hilbertKey     = iHilbert<KeyType>(L1Range * xi + L1Range - 1, L1Range * yi + L1Range - 1,
                                                               L1Range * zi + L1Range - 1);
                    unsigned hilbertOctant = octalDigit(hilbertKey, 1);
                    EXPECT_EQ(mortonOctant, hilbertToMorton[hilbertOctant]);
                }
            }
        }
    }
}

TEST(HilbertCode, firstOrderCurve)
{
    firstOrderCurve<unsigned>();
    firstOrderCurve<uint64_t>();
}

//! @brief verifies continuity properties across consecutive octants at all levels
template<class KeyType>
void continuityTest()
{
    for (unsigned level = 1; level < maxTreeLevel<KeyType>{}; ++level)
    {
        // on the highest level, we can only check 7 octant crossings
        int maxOctant = (level > 1) ? 8 : 7;

        for (int octant = 0; octant < maxOctant; ++octant)
        {
            KeyType lastKey      = (octant + 1) * nodeRange<KeyType>(level) - 1;
            KeyType firstNextKey = lastKey + 1;

            auto [x, y, z] = decodeHilbert(lastKey);

            auto [xnext, ynext, znext] = decodeHilbert(firstNextKey);

            // the points in 3D space should be right next to each other, i.e. delta == 1
            // this is a property that the Z-curve does not have
            int delta = std::abs(int(x) - int(xnext)) + std::abs(int(y) - int(ynext)) + std::abs(int(z) - int(znext));

            EXPECT_EQ(delta, 1);
        }
    }
}

TEST(HilbertCode, continuity)
{
    continuityTest<unsigned>();
    continuityTest<uint64_t>();
}

//! @brief tests numKeys random 3D points for encoding/decoding consistency
template<class KeyType>
void inversionTest()
{
    int numKeys  = 1000;
    int maxCoord = (1 << maxTreeLevel<KeyType>{}) - 1;

    std::mt19937 gen;
    std::uniform_int_distribution<unsigned> distribution(0, maxCoord);

    auto getRand = [&distribution, &gen]() { return distribution(gen); };

    std::vector<unsigned> x(numKeys);
    std::vector<unsigned> y(numKeys);
    std::vector<unsigned> z(numKeys);

    std::generate(begin(x), end(x), getRand);
    std::generate(begin(y), end(y), getRand);
    std::generate(begin(z), end(z), getRand);

    for (int i = 0; i < numKeys; ++i)
    {
        KeyType hilbertKey = iHilbert<KeyType>(x[i], y[i], z[i]);

        auto [a, b, c] = decodeHilbert(hilbertKey);
        EXPECT_EQ(x[i], a);
        EXPECT_EQ(y[i], b);
        EXPECT_EQ(z[i], c);
    }
}

TEST(HilbertCode, inversion)
{
    inversionTest<unsigned>();
    inversionTest<uint64_t>();
}

template<class KeyType>
std::tuple<KeyType, KeyType> findMinMaxKey(const IBox& ibox)
{
    std::vector<KeyType> cornerKeys;

    int cubeLength = ibox.xmax() - ibox.xmin() - 1;
    for (int hx = 0; hx < 2; ++hx)
        for (int hy = 0; hy < 2; ++hy)
            for (int hz = 0; hz < 2; ++hz)
            {
                cornerKeys.push_back(iHilbert<KeyType>(ibox.xmin() + hx * cubeLength, ibox.ymin() + hy * cubeLength,
                                                       ibox.zmin() + hz * cubeLength));
            }

    return {*std::min_element(cornerKeys.begin(), cornerKeys.end()),
            *std::max_element(cornerKeys.begin(), cornerKeys.end()) + 1};
}

template<class KeyType>
void makeHilbertIBox()
{
    {
        constexpr unsigned cubeLength = 1u << maxTreeLevel<KeyType>{};
        KeyType start                 = pad(KeyType(03), 3);
        KeyType end                   = pad(KeyType(04), 3);

        IBox box = hilbertIBox(start, treeLevel(end - start));

        IBox reference(0, cubeLength / 2, cubeLength / 2, cubeLength, 0, cubeLength / 2);

        EXPECT_EQ(box, reference);
    }

    int level = 4;
    {
        int maxL = 1 << maxTreeLevel<KeyType>{};
        int L    = 1 << (maxTreeLevel<KeyType>{} - level);

        for (int ix = 0; ix < maxL; ix += L)
            for (int iy = 0; iy < maxL; iy += L)
                for (int iz = 0; iz < maxL; iz += L)
                {
                    {
                        IBox reference(ix, ix + L, iy, iy + L, iz, iz + L);
                        auto [start, end] = findMinMaxKey<KeyType>(reference);
                        IBox testBox      = hilbertIBox(start, treeLevel(end - start));
                        EXPECT_EQ(testBox, reference);
                    }
                    {
                        IBox reference(ix, ix + 1, iy, iy + 1, iz, iz + 1);
                        auto [start, end] = findMinMaxKey<KeyType>(reference);
                        EXPECT_EQ(start + 1, end);

                        IBox testBox = hilbertIBox(start, treeLevel(end - start));
                        EXPECT_EQ(testBox, reference);
                    }
                }
    }
}

TEST(HilbertCode, makeIBox)
{
    makeHilbertIBox<unsigned>();
    makeHilbertIBox<uint64_t>();
}

template<class KeyType>
void hilbertNeighbor()
{
    using Integer = typename KeyType::ValueType;
    {
        unsigned level = 1;
        int coordRange = 1u << (maxTreeLevel<KeyType>{} - level);
        IBox ibox(0, coordRange, 0, coordRange, 0, coordRange);
        EXPECT_EQ(sfcNeighbor<KeyType>(ibox, level, 0, 0, 0), 0);
    }
    {
        unsigned level = 1;
        int coordRange = 1u << (maxTreeLevel<KeyType>{} - level);
        IBox ibox(0, coordRange, 0, coordRange, 0, coordRange);
        EXPECT_EQ(sfcNeighbor<KeyType>(ibox, level, 0, 1, 1), pad(Integer(02), 3));
    }
    {
        unsigned level = 1;
        int coordRange = 1u << (maxTreeLevel<KeyType>{} - level);
        IBox ibox(0, coordRange, 0, coordRange, 0, coordRange);
        EXPECT_EQ(sfcNeighbor<KeyType>(ibox, level, 0, 1, 0), pad(Integer(03), 3));
    }
    {
        unsigned level = 1;
        int coordRange = 1u << (maxTreeLevel<KeyType>{} - level);
        IBox ibox(0, coordRange, 0, coordRange, 0, coordRange);
        EXPECT_EQ(sfcNeighbor<KeyType>(ibox, level, 1, 1, 1), pad(Integer(05), 3));
    }
    {
        unsigned level = 1;
        int coordRange = 1u << (maxTreeLevel<KeyType>{} - level);
        IBox ibox(0, coordRange, 0, coordRange, 0, coordRange);
        EXPECT_EQ(sfcNeighbor<KeyType>(ibox, level, -1, -1, -1), pad(Integer(05), 3));
    }
}

TEST(HilbertCode, neighbor)
{
    hilbertNeighbor<HilbertKey<unsigned>>();
    hilbertNeighbor<HilbertKey<uint64_t>>();
}
