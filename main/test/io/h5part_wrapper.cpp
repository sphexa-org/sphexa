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
 * @brief Unit tests for H5Part C++ wrappers
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <iostream>
#include <filesystem>

#include "gtest/gtest.h"

#include "io/h5part_wrapper.hpp"

using namespace sphexa::fileutils;

TEST(H5PartCpp, typesafeStepAttrRead)
{
    std::string testfile = "variant_read.h5";
    if (std::filesystem::exists(testfile)) { std::filesystem::remove(testfile); }

    {
        h5_file_t h5File = H5OpenFile(testfile.c_str(), H5_O_RDWR, H5_PROP_DEFAULT);
        H5SetStep(h5File, 0);

        double float64Attr = 0.5;
        H5WriteStepAttribT(h5File, "float64Attr", &float64Attr, 1);
        int64_t int64Attr = 42;
        H5WriteStepAttribT(h5File, "int64Attr", &int64Attr, 1);
        uint64_t uint64Attr = uint64_t(2) << 40;
        H5WriteStepAttribT(h5File, "uint64Attr", &uint64Attr, 1);
        char int8Attr = 1;
        H5WriteStepAttribT(h5File, "int8Attr", &int8Attr, 1);

        H5CloseFile(h5File);
    }
    {
        h5_file_t h5File = H5OpenFile(testfile.c_str(), H5_O_RDWR, H5_PROP_DEFAULT);
        H5SetStep(h5File, 0);

        {
            std::vector<double> a(1);
            readH5PartStepAttribute(a.data(), a.size(), 0, h5File);
            EXPECT_EQ(a[0], 0.5);
        }
        {
            std::vector<float> a(1);
            readH5PartStepAttribute(a.data(), a.size(), 0, h5File);
            EXPECT_EQ(a[0], 0.5);
        }
        {
            std::vector<int> a(1);
            EXPECT_THROW(readH5PartStepAttribute(a.data(), a.size(), 0, h5File), std::runtime_error);
        }
        {
            std::vector<int64_t> a(1);
            readH5PartStepAttribute(a.data(), a.size(), 1, h5File);
            EXPECT_EQ(a[0], 42);
        }
        {
            std::vector<double> a(1);
            readH5PartStepAttribute(a.data(), a.size(), 1, h5File);
            EXPECT_EQ(a[0], 42);
        }
        {
            std::vector<int> a(1);
            EXPECT_THROW(readH5PartStepAttribute(a.data(), a.size(), 1, h5File), std::runtime_error);
        }
        {
            std::vector<uint64_t> a(1);
            readH5PartStepAttribute(a.data(), a.size(), 2, h5File);
            EXPECT_EQ(a[0], uint64_t(2) << 40);
        }
        {
            std::vector<char> a(1);
            readH5PartStepAttribute(a.data(), a.size(), 3, h5File);
            EXPECT_EQ(a[0], 1);
        }

        H5CloseFile(h5File);
    }
}

TEST(H5PartCpp, typesafeFileAttrRead)
{
    std::string testfile = "variant_read.h5";
    if (std::filesystem::exists(testfile)) { std::filesystem::remove(testfile); }

    {
        h5_file_t h5File = H5OpenFile(testfile.c_str(), H5_O_WRONLY, H5_PROP_DEFAULT);

        double float64Attr = 0.5;
        H5WriteFileAttribT(h5File, "float64Attr", &float64Attr, 1);
        int64_t int64Attr = 42;
        H5WriteFileAttribT(h5File, "int64Attr", &int64Attr, 1);
        uint64_t uint64Attr = uint64_t(2) << 40;
        H5WriteFileAttribT(h5File, "uint64Attr", &uint64Attr, 1);
        char int8Attr = 1;
        H5WriteFileAttribT(h5File, "int8Attr", &int8Attr, 1);

        H5CloseFile(h5File);
    }
    {
        h5_file_t h5File = H5OpenFile(testfile.c_str(), H5_O_RDONLY, H5_PROP_DEFAULT);

        {
            std::vector<double> a(1);
            readH5PartFileAttribute(a.data(), a.size(), 0, h5File);
            EXPECT_EQ(a[0], 0.5);
        }
        {
            std::vector<float> a(1);
            readH5PartFileAttribute(a.data(), a.size(), 0, h5File);
            EXPECT_EQ(a[0], 0.5);
        }
        {
            std::vector<int> a(1);
            EXPECT_THROW(readH5PartFileAttribute(a.data(), a.size(), 0, h5File), std::runtime_error);
        }
        {
            std::vector<int64_t> a(1);
            readH5PartFileAttribute(a.data(), a.size(), 1, h5File);
            EXPECT_EQ(a[0], 42);
        }
        {
            std::vector<double> a(1);
            readH5PartFileAttribute(a.data(), a.size(), 1, h5File);
            EXPECT_EQ(a[0], 42);
        }
        {
            std::vector<int> a(1);
            EXPECT_THROW(readH5PartFileAttribute(a.data(), a.size(), 1, h5File), std::runtime_error);
        }
        {
            std::vector<uint64_t> a(1);
            readH5PartFileAttribute(a.data(), a.size(), 2, h5File);
            EXPECT_EQ(a[0], uint64_t(2) << 40);
        }
        {
            std::vector<char> a(1);
            readH5PartFileAttribute(a.data(), a.size(), 3, h5File);
            EXPECT_EQ(a[0], 1);
        }

        H5CloseFile(h5File);
    }
}
