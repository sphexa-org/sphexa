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
 * @brief turbulence stirring tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <memory>
#include <random>
#include <sstream>

#include "gtest/gtest.h"

TEST(Turbulence, rngSerialize)
{
    std::mt19937 engine;
    engine.seed(42);

    const auto originalEngine = engine;
    // serialize originalEngine into a string
    std::stringstream s;
    s << originalEngine;
    std::string engineState = s.str();

    EXPECT_EQ(originalEngine, engine);

    for (int i = 0; i < 100; ++i)
    {
        engine();
    }

    // engines are now in a different state
    EXPECT_NE(originalEngine, engine);

    // check conversion from char array, as that's what's going to be read from HDF5
    auto stateChar = std::make_unique<char[]>(engineState.size());
    std::copy(engineState.begin(), engineState.end(), stateChar.get());

    std::stringstream t;
    t << stateChar.get();
    t >> engine;

    EXPECT_EQ(originalEngine, engine);
}
