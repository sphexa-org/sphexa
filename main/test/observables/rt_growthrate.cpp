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

#include "gtest/gtest.h"
#include "observables/gpu_reductions.h"

using namespace sphexa;
using T = double;

TEST(rt_growthrate, remove_functor)
{
    std::vector<AuxT<T>> test(5);
    T testPos = 2.0;
    test[1].pos = testPos;
    test[1].vel = 0.5;
    ASSERT_TRUE(std::isnan(test[0].pos));
    auto newEnd = std::remove_if(test.begin(), test.end(), invalidAuxTEntry<T>{});
    ASSERT_EQ(newEnd, test.begin() + 1);
    ASSERT_EQ(testPos, test[0].pos);
}

TEST(rt_growthrate, sort_functors)
{
    int n = 5;
    std::vector<AuxT<T>> up(n);
    std::vector<AuxT<T>> down(n);
    for (int i = 0; i < n; ++i)
    {
        up[i].pos = double(i);
        down[i].pos = double(i);
    }
    std::sort(up.begin(), up.end(), greaterRT{});
    std::sort(down.begin(), down.end(), lowerRT{});
    for (int i = 0; i < n; ++i)
    {
        EXPECT_TRUE(up[i].pos == down[n-i-1].pos);
    }
}