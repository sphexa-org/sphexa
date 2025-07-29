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
 * @brief Unit tests for I/O related functionality
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <iostream>
#include "gtest/gtest.h"
#include "io/arg_parser.hpp"

using namespace sphexa;

TEST(IO, parseCommaListEmpy)
{
    constexpr int            argc       = 3;
    const char*              argv[argc] = {"-f", "-w", "1"};
    ArgParser                parser(argc, argv);
    std::vector<std::string> ref;
    EXPECT_EQ(parser.getCommaList("-f"), ref);
}

TEST(IO, strIsIntegral)
{
    EXPECT_TRUE(strIsIntegral("42"));
    EXPECT_TRUE(strIsIntegral("-42"));

    EXPECT_FALSE(strIsIntegral("3.1"));
    EXPECT_FALSE(strIsIntegral("3a"));
}

TEST(IO, isExtraOutputStep)
{
    std::vector<std::string> writeExtra{"1", "4.2", "5", "0.77"};

    EXPECT_TRUE(isExtraOutputStep(1, 0, 0, writeExtra));
    EXPECT_TRUE(isExtraOutputStep(0, 4.19, 4.21, writeExtra));
    EXPECT_TRUE(isExtraOutputStep(0, 4.2, 4.21, writeExtra));
    EXPECT_TRUE(isExtraOutputStep(5, 0, 0, writeExtra));
    EXPECT_TRUE(isExtraOutputStep(5, 0.76, 0.78, writeExtra));

    EXPECT_FALSE(isExtraOutputStep(4, 0, 0, writeExtra));
    EXPECT_FALSE(isExtraOutputStep(0, 4.19, 4.2, writeExtra));
    EXPECT_FALSE(isExtraOutputStep(6, 5.19, 6.0, writeExtra));
}

TEST(IO, isOutputTime)
{
    EXPECT_FALSE(isOutputTime(9.9, 10.1, "2"));
    EXPECT_TRUE(isOutputTime(9.9, 10.1, "2.0"));
    EXPECT_FALSE(isOutputTime(10.01, 10.1, "2.0"));
    EXPECT_FALSE(isOutputTime(0.0, 1.1e-6, "0.01"));
}

TEST(IO, isOutputStep)
{
    EXPECT_FALSE(isOutputStep(42, "-1"));
    EXPECT_FALSE(isOutputStep(42, "0"));
    EXPECT_TRUE(isOutputStep(42, "42"));
    EXPECT_TRUE(isOutputStep(84, "42"));
    EXPECT_FALSE(isOutputStep(42, "42.0"));
}

TEST(IO, numberAfterSign)
{
    EXPECT_EQ(numberAfterSign("chkp.h5", ","), -1);
    EXPECT_EQ(numberAfterSign("chkp.h5,1", ","), 1);
    EXPECT_EQ(numberAfterSign("chkp.h5,42", ","), 42);
    EXPECT_EQ(numberAfterSign("chkp.h5,42,42", ","), -1);

    EXPECT_EQ(numberAfterSign("chkp.h5-O_O-42", "-O_O-"), 42);
}