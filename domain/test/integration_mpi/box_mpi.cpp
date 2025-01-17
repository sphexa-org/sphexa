/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
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
    T val = rank + 1;
    std::vector<T> x{-val, val};
    std::vector<T> y{val, 2 * val};
    std::vector<T> z{-val, -2 * val};

    Box<T> box = makeGlobalBox(x.data(), y.data(), z.data(), x.size(), Box<T>{0, 1});

    T rVal = numRanks;
    EXPECT_EQ(box.xmin(), -rVal);
    EXPECT_EQ(box.xmax(), rVal);
    EXPECT_EQ(box.ymin(), T(1));
    EXPECT_EQ(box.ymax(), 2 * rVal);
    EXPECT_EQ(box.zmin(), -2 * rVal);
    EXPECT_EQ(box.zmax(), T(-1));

    auto open     = BoundaryType::open;
    auto periodic = BoundaryType::periodic;

    // PBC case
    {
        Box<T> pbcBox{0, 1, 0, 1, 0, 1, periodic, periodic, periodic};
        Box<T> newPbcBox = makeGlobalBox(x.data(), y.data(), z.data(), x.size(), pbcBox);
        EXPECT_EQ(pbcBox, newPbcBox);
    }
    // partial PBC
    {
        Box<T> pbcBox{0, 1, 0, 1, 0, 1, open, periodic, periodic};
        Box<T> newPbcBox = makeGlobalBox(x.data(), y.data(), z.data(), x.size(), pbcBox);
        Box<T> refBox{-rVal, rVal, 0, 1, 0, 1, open, periodic, periodic};
        EXPECT_EQ(refBox, newPbcBox);
    }
    {
        Box<T> pbcBox{0, 1, 0, 1, 0, 1, periodic, open, periodic};
        Box<T> newPbcBox = makeGlobalBox(x.data(), y.data(), z.data(), x.size(), pbcBox);
        Box<T> refBox{0, 1, T(1), 2 * rVal, 0, 1, periodic, open, periodic};
        EXPECT_EQ(refBox, newPbcBox);
    }
    {
        Box<T> pbcBox{0, 1, 0, 1, 0, 1, periodic, periodic, open};
        Box<T> newPbcBox = makeGlobalBox(x.data(), y.data(), z.data(), x.size(), pbcBox);
        Box<T> refBox{0, 1, 0, 1, -2 * rVal, T(-1), periodic, periodic, open};
        EXPECT_EQ(refBox, newPbcBox);
    }
}
