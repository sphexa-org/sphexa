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
 * @brief Unit tests for grid-based initialization functionality
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <algorithm>

#include "gtest/gtest.h"

#include "cstone/sfc/common.hpp"

#include "init/grid.hpp"

using namespace sphexa;

TEST(Grids, intersect)
{
    using T = double;
    cstone::FBox<T> box{0.12, 0.50, 0.26, 0.44, 0.55, 0.8};

    cstone::Vec3<int> multiplicity = {1, 2, 3};
    auto [l, u]                    = gridIntersection(box, multiplicity);

    cstone::Vec3<int> refLower{0, 0, 1};
    cstone::Vec3<int> refUpper{1, 1, 3};

    EXPECT_EQ(l, refLower);
    EXPECT_EQ(u, refUpper);
}

TEST(Grids, scaleToGlobal)
{
    using T = double;
    cstone::Box<T> box{-1, 1};

    cstone::Vec3<int> multiplicity = {4, 4, 4};

    {
        cstone::Vec3<T> testX{0.0, 0.0, 0.0};
        auto            scaledX = scaleBlockToGlobal(testX, {0, 0, 0}, multiplicity, box);

        EXPECT_NEAR(scaledX[0], box.xmin(), 1e-10);
        EXPECT_NEAR(scaledX[1], box.ymin(), 1e-10);
        EXPECT_NEAR(scaledX[2], box.zmin(), 1e-10);
    }
    {
        cstone::Vec3<T> testX{0.0, 0.0, 0.0};
        auto            scaledX = scaleBlockToGlobal(testX, {2, 2, 2}, multiplicity, box);

        EXPECT_NEAR(scaledX[0], 0.0, 1e-10);
        EXPECT_NEAR(scaledX[1], 0.0, 1e-10);
        EXPECT_NEAR(scaledX[2], 0.0, 1e-10);
    }
    {
        cstone::Vec3<T> testX{1.0, 1.0, 1.0};
        auto            scaledX = scaleBlockToGlobal(testX, {3, 3, 3}, multiplicity, box);

        EXPECT_NEAR(scaledX[0], box.xmax(), 1e-10);
        EXPECT_NEAR(scaledX[1], box.ymax(), 1e-10);
        EXPECT_NEAR(scaledX[2], box.zmax(), 1e-10);
    }
    {
        cstone::Vec3<T> testX{0.1, 0.5, 0.6};
        auto            scaledX = scaleBlockToGlobal(testX, {0, 1, 2}, multiplicity, box);

        EXPECT_NEAR(scaledX[0], -0.95, 1e-10);
        EXPECT_NEAR(scaledX[1], -0.25, 1e-10);
        EXPECT_NEAR(scaledX[2], 0.3, 1e-10);
    }
}

TEST(Grids, assembleCuboid)
{
    using T       = double;
    using KeyType = unsigned;
    cstone::Box<T> box{-1, 1};

    cstone::Vec3<int> multiplicity = {1, 3, 4};

    std::vector<T> xb{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    std::vector<T> yb{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    std::vector<T> zb{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    // 3 SFC segments: 0-k1 k1-k2 k2-nodeRange(0)
    KeyType k1 = 01234567012;
    KeyType k2 = 05123456701;

    std::vector<T> x1, y1, z1;
    assembleCuboid<T>(KeyType(0), k1, box, multiplicity, xb, yb, zb, x1, y1, z1);

    std::vector<T> x2, y2, z2;
    assembleCuboid<T>(k1, k2, box, multiplicity, xb, yb, zb, x2, y2, z2);

    std::vector<T> x3, y3, z3;
    assembleCuboid<T>(k2, cstone::nodeRange<KeyType>(0), box, multiplicity, xb, yb, zb, x3, y3, z3);

    // total number of particles in the 3 segments together should be initBlock.size() * multiplicity^3
    size_t totalSize = x1.size() + x2.size() + x3.size();
    EXPECT_EQ(totalSize, multiplicity[0] * multiplicity[1] * multiplicity[2] * xb.size());

    std::vector<KeyType> keys(totalSize);
    auto                 ksfc = cstone::sfcKindPointer(keys.data());

    cstone::computeSfcKeys(x1.data(), y1.data(), z1.data(), ksfc, x1.size(), box);
    cstone::computeSfcKeys(x2.data(), y2.data(), z2.data(), ksfc + x1.size(), x2.size(), box);
    cstone::computeSfcKeys(x3.data(), y3.data(), z3.data(), ksfc + x1.size() + x2.size(), x3.size(), box);

    for (size_t i = 0; i < x1.size(); ++i)
    {
        EXPECT_LT(keys[i], k1);
    }
    for (size_t i = x1.size(); i < x1.size() + x2.size(); ++i)
    {
        EXPECT_LE(k1, keys[i]);
        EXPECT_LT(keys[i], k2);
    }
    for (size_t i = x1.size() + x2.size(); i < keys.size(); ++i)
    {
        EXPECT_LE(k2, keys[i]);
    }

    std::sort(keys.begin(), keys.end());
    auto uit = std::unique(keys.begin(), keys.end());

    // combined particles from duplicates should not contain any duplicate particles
    EXPECT_EQ(uit, keys.end());

    // explicit construction
    std::vector<T>  X, Y, Z;
    cstone::Vec3<T> frag{box.lx() / multiplicity[0], box.ly() / multiplicity[1], box.lz() / multiplicity[2]};
    for (int i = 0; i < multiplicity[0]; i++)
    {
        for (int j = 0; j < multiplicity[1]; ++j)
        {
            for (int k = 0; k < multiplicity[2]; ++k)
            {
                auto selectBox = cstone::FBox<T>(box.xmin() + i * frag[0], box.xmin() + (i + 1) * frag[0],
                                                 box.ymin() + j * frag[1], box.ymin() + (j + 1) * frag[1],
                                                 box.zmin() + k * frag[2], box.zmin() + (k + 1) * frag[2]);
                auto np        = countSelection<T>(selectBox, box, {i, j, k}, multiplicity, xb, yb, zb);
                X.resize(X.size() + np);
                Y.resize(Y.size() + np);
                Z.resize(Z.size() + np);
                extractBlock<T>(selectBox, box, {i, j, k}, multiplicity, xb, yb, zb,
                                std::span(X).subspan(X.size() - np), std::span(Y).subspan(Y.size() - np),
                                std::span(Z).subspan(Z.size() - np));
            }
        }
    }

    std::vector<KeyType> keys2(X.size());
    auto                 ksfc2 = cstone::sfcKindPointer(keys2.data());
    cstone::computeSfcKeys(X.data(), Y.data(), Z.data(), ksfc2, X.size(), box);

    std::sort(keys2.begin(), keys2.end());

    for (size_t i = 0; i < keys2.size(); ++i)
    {
        EXPECT_EQ(keys[i], keys2[i]);
    }
}

TEST(IsobaricCube, pyramidStretch)
{
    using T = double;

    T h = 0.25;
    T s = 0.40548013;
    T r = 0.5;

    {
        cstone::Vec3<T> X{s, 0, 0};
        auto            Xp = X * cappedPyramidStretch(X, h, s, r);
        EXPECT_NEAR(Xp[0], h, 1e-6);
        EXPECT_NEAR(Xp[1], 0, 1e-6);
        EXPECT_NEAR(Xp[2], 0, 1e-6);
    }
    { // point on the surface of the stretch boundary stays
        cstone::Vec3<T> X{s, 0.1, 0.2};
        auto            Xp = X * cappedPyramidStretch(X, h, s, r);
        EXPECT_NEAR(Xp[0], h, 1e-6);
        EXPECT_LT(Xp[1], X[1]);
        EXPECT_LT(Xp[2], X[2]);
    }
    { // point on surface of stretch-boundary gets mapped to the surface of the internal cube
        cstone::Vec3<T> X{s, s, s};
        auto            Xp = X * cappedPyramidStretch(X, h, s, r);
        EXPECT_NEAR(Xp[0], h, 1e-6);
        EXPECT_NEAR(Xp[1], h, 1e-6);
        EXPECT_NEAR(Xp[2], h, 1e-6);
    }
    { // point on the surface of the outer cube stays on the outer surface
        cstone::Vec3<T> X{r, r, r};
        auto            Xp = X * cappedPyramidStretch(X, h, s, r);
        EXPECT_NEAR(Xp[0], r, 1e-6);
        EXPECT_NEAR(Xp[1], r, 1e-6);
        EXPECT_NEAR(Xp[2], r, 1e-6);
    }
    { // point on the surface of the outer cube stays on the outer surface
        cstone::Vec3<T> X{r, 0.1, 0.2};
        auto            Xp = X * cappedPyramidStretch(X, h, s, r);
        EXPECT_NEAR(Xp[0], r, 1e-6);
        EXPECT_NEAR(Xp[1], 0.1, 1e-6);
        EXPECT_NEAR(Xp[2], 0.2, 1e-6);
    }
}

TEST(IsobaricCube, stretchFactor)
{
    using T = double;

    T h        = 0.25;
    T r        = 0.5;
    T rhoRatio = 8;

    T s = computeStretchFactor(h, r, rhoRatio);

    T sRef = 0.40548013;
    EXPECT_NEAR(s, sRef, 1e-6);
}
