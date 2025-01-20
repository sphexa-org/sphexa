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
 * @brief Halo exchange test
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <gtest/gtest.h>

#include "cstone/halos/exchange_halos.hpp"

using namespace cstone;

void simpleTest(int thisRank)
{
    int nRanks = 2;
    std::vector<int> nodeList{0, 1, 10, 11};
    std::vector<int> offsets{0, 1, 3, 6, 10};

    int localCount, localOffset;
    if (thisRank == 0)
    {
        localCount  = 3;
        localOffset = 0;
    }
    if (thisRank == 1)
    {
        localCount  = 7;
        localOffset = 3;
    }

    RecvList incomingHalos(nRanks);
    SendList outgoingHalos(nRanks);

    if (thisRank == 0)
    {
        incomingHalos[1] = {3, 10};
        outgoingHalos[1].addRange(0, 1);
        outgoingHalos[1].addRange(1, 3);
    }
    if (thisRank == 1)
    {
        incomingHalos[0] = {0, 3};
        outgoingHalos[0].addRange(3, 6);
        outgoingHalos[0].addRange(6, 10);
    }

    std::vector<double> x(*offsets.rbegin());
    std::vector<float> y(*offsets.rbegin());
    std::vector<util::array<int, 3>> velocity(*offsets.rbegin());

    int xshift = 20;
    int yshift = 30;
    for (int i = 0; i < localCount; ++i)
    {
        int eoff       = localOffset + i;
        x[eoff]        = eoff + xshift;
        y[eoff]        = eoff + yshift;
        velocity[eoff] = util::array<int, 3>{eoff, eoff + 1, eoff + 2};
    }

    if (thisRank == 0)
    {
        std::vector<double> xOrig{20, 21, 22, 0, 0, 0, 0, 0, 0, 0};
        std::vector<float> yOrig{30, 31, 32, 0, 0, 0, 0, 0, 0, 0};
        EXPECT_EQ(xOrig, x);
        EXPECT_EQ(yOrig, y);
    }
    if (thisRank == 1)
    {
        std::vector<double> xOrig{0, 0, 0, 23, 24, 25, 26, 27, 28, 29};
        std::vector<float> yOrig{0, 0, 0, 33, 34, 35, 36, 37, 38, 39};
        EXPECT_EQ(xOrig, x);
        EXPECT_EQ(yOrig, y);
    }

    haloexchange(0, incomingHalos, outgoingHalos, x.data(), y.data(), velocity.data());

    std::vector<double> xRef{20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
    std::vector<float> yRef{30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
    std::vector<util::array<int, 3>> velocityRef{{0, 1, 2}, {1, 2, 3}, {2, 3, 4}, {3, 4, 5},  {4, 5, 6},
                                                 {5, 6, 7}, {6, 7, 8}, {7, 8, 9}, {8, 9, 10}, {9, 10, 11}};
    EXPECT_EQ(xRef, x);
    EXPECT_EQ(yRef, y);
    EXPECT_EQ(velocityRef, velocity);
}

TEST(HaloExchange, simpleTest)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    constexpr int thisExampleRanks = 2;

    if (nRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    simpleTest(rank);
}