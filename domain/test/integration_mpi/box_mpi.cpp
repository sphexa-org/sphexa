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
 * @brief Tests the global bounding box
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include <numeric>
#include <random>
#include <vector>

#include "cstone/sfc/box_mpi.hpp"

using namespace cstone;

TEST(GlobalBox, localMinMax)
{
    using T = double;

    int numElements = 1000;
    std::vector<T> x(numElements);
    std::iota(begin(x), end(x), 1);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(begin(x), end(x), g);

    auto [gmin, gmax] = MinMax<T>{}(x.data(), x.data() + x.size());
    EXPECT_EQ(gmin, T(1));
    EXPECT_EQ(gmax, T(numElements));
}

template<class T>
void makeGlobalBox(int rank, int numRanks)
{
    constexpr T infinity      = std::numeric_limits<T>::infinity();
    constexpr T shrink_factor = 0.1;

    T val = rank + 1;
    std::vector<T> x{-val, val};
    std::vector<T> y{val, 2 * val};
    std::vector<T> z{-val, -2 * val};

    Box<T> box = makeGlobalBox(x.data(), y.data(), z.data(), x.size(), Box<T>{0, 1}, infinity);

    T rVal = numRanks;
    EXPECT_EQ(box.xmin(), -rVal);
    EXPECT_EQ(box.xmax(), rVal);
    EXPECT_EQ(box.ymin(), T(1));
    EXPECT_EQ(box.ymax(), 2 * rVal);
    EXPECT_EQ(box.zmin(), -2 * rVal);
    EXPECT_EQ(box.zmax(), T(-1));

    auto open     = BoundaryType::open;
    auto periodic = BoundaryType::periodic;

    auto getDoubleLengthBox = [](const auto& box)
    {
        const std::array<T, 3> box_centre{(box.xmax() + box.xmin()) / T{2.}, (box.ymax() + box.ymin()) / T{2.},
                                          (box.zmax() + box.zmin()) / T{2.}};

        const Box<T> doubleLengthBox(box_centre[0] - box.lx(), box_centre[0] + box.lx(), box_centre[1] - box.ly(),
                                     box_centre[1] + box.ly(), box_centre[2] - box.lz(), box_centre[2] + box.lz(),
                                     box.boundaryX(), box.boundaryY(), box.boundaryZ());
        return doubleLengthBox;
    };

    auto contains = [](const auto& large_box, const auto& small_box)
    {
        return (large_box.xmin() <= small_box.xmin() && large_box.ymin() <= small_box.ymin() &&
                large_box.zmin() <= small_box.zmin() && large_box.xmax() >= small_box.xmax() &&
                large_box.ymax() >= small_box.ymax() && large_box.zmax() >= small_box.zmax());
    };

    // PBC case
    {
        Box<T> pbcBox{0, 1, 0, 1, 0, 1, periodic, periodic, periodic};
        Box<T> newPbcBox = makeGlobalBox(x.data(), y.data(), z.data(), x.size(), pbcBox, infinity);
        EXPECT_EQ(pbcBox, newPbcBox);
    }
    // partial PBC
    {
        Box<T> pbcBox{0, 1, 0, 1, 0, 1, open, periodic, periodic};
        Box<T> newPbcBox = makeGlobalBox(x.data(), y.data(), z.data(), x.size(), pbcBox, infinity);
        Box<T> refBox{-rVal, rVal, 0, 1, 0, 1, open, periodic, periodic};
        EXPECT_EQ(refBox, newPbcBox);

        const auto doubleBox = getDoubleLengthBox(newPbcBox);
        Box<T> shrinkedBox   = makeGlobalBox(x.data(), y.data(), z.data(), x.size(), doubleBox, shrink_factor);

        EXPECT_NEAR(shrinkedBox.lx(), (1. - 2. * shrink_factor) * doubleBox.lx(), 1e-6);
        EXPECT_EQ(shrinkedBox.ly(), doubleBox.ly()); //, 1e-6);
        EXPECT_EQ(shrinkedBox.lz(), doubleBox.lz());

        EXPECT_TRUE(contains(shrinkedBox, newPbcBox));
    }
    {
        Box<T> pbcBox{0, 1, 0, 1, 0, 1, periodic, open, periodic};
        Box<T> newPbcBox = makeGlobalBox(x.data(), y.data(), z.data(), x.size(), pbcBox, infinity);
        Box<T> refBox{0, 1, T(1), 2 * rVal, 0, 1, periodic, open, periodic};
        EXPECT_EQ(refBox, newPbcBox);

        const auto doubleBox = getDoubleLengthBox(newPbcBox);
        Box<T> shrinkedBox   = makeGlobalBox(x.data(), y.data(), z.data(), x.size(), doubleBox, shrink_factor);

        EXPECT_EQ(shrinkedBox.lx(), doubleBox.lx());
        EXPECT_NEAR(shrinkedBox.ly(), (1. - 2. * shrink_factor) * doubleBox.ly(), 1e-6);
        EXPECT_EQ(shrinkedBox.lz(), doubleBox.lz());
        EXPECT_TRUE(contains(shrinkedBox, newPbcBox));
    }
    {
        Box<T> pbcBox{0, 1, 0, 1, 0, 1, periodic, periodic, open};
        Box<T> newPbcBox = makeGlobalBox(x.data(), y.data(), z.data(), x.size(), pbcBox, infinity);
        Box<T> refBox{0, 1, 0, 1, -2 * rVal, T(-1), periodic, periodic, open};
        EXPECT_EQ(refBox, newPbcBox);

        const auto doubleBox = getDoubleLengthBox(newPbcBox);
        Box<T> shrinkedBox   = makeGlobalBox(x.data(), y.data(), z.data(), x.size(), doubleBox, shrink_factor);

        EXPECT_EQ(shrinkedBox.lx(), doubleBox.lx());
        EXPECT_EQ(shrinkedBox.ly(), doubleBox.ly());
        EXPECT_NEAR(shrinkedBox.lz(), (1. - 2. * shrink_factor) * doubleBox.lz(), 1e-6);
        EXPECT_TRUE(contains(shrinkedBox, newPbcBox));
    }
}

TEST(GlobalBox, makeGlobalBox)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    makeGlobalBox<float>(rank, numRanks);
    makeGlobalBox<double>(rank, numRanks);
}
