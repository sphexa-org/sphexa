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
 * @brief Buffer description tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/util/pack_buffers.hpp"

using namespace util;

TEST(PackBuffers, computeByteOffsets1)
{
    constexpr size_t alignment = 128;
    size_t sendCount           = 1001;

    float* p1 = nullptr;
    char* p2  = nullptr;
    long* p3   = nullptr;

    auto offsets = computeByteOffsets(sendCount, alignment, p1, p2, p3);

    EXPECT_EQ(offsets[0], 0);
    EXPECT_EQ(offsets[1], 4096);
    EXPECT_EQ(offsets[2], 4096 + 1024);
    EXPECT_EQ(offsets[3], 4096 + 1024 + 8064);
}

TEST(PackBuffers, computeByteOffsetsPadLast)
{
    constexpr size_t alignment = 8;
    size_t sendCount           = 1;

    double* p1 = nullptr;
    long* p2   = nullptr;
    float* p3  = nullptr;

    auto offsets = computeByteOffsets(sendCount, alignment, p1, p2, p3);
    EXPECT_EQ(offsets[0], 0);
    EXPECT_EQ(offsets[1], 8);
    EXPECT_EQ(offsets[2], 16);
    EXPECT_EQ(offsets[3], 24);
}

TEST(PackBuffers, packBufferPtrsA1)
{
    constexpr int Alignment = 1;
    char* packedBufferBase  = 0; // NOLINT

    size_t bufferSizes = 10;
    auto* p1           = reinterpret_cast<double*>(1024);
    auto* p2           = reinterpret_cast<char*>(2048);
    auto* p3           = reinterpret_cast<util::array<int, 4>*>(4096);
    auto* p4           = reinterpret_cast<int*>(8192);

    auto packed = packBufferPtrs<Alignment>(packedBufferBase, bufferSizes, p1, p2, p3, p4);

    EXPECT_EQ(reinterpret_cast<long>(std::get<0>(packed)[0]), 1024);
    EXPECT_EQ(reinterpret_cast<long>(std::get<0>(packed)[1]), 0);
    static_assert(std::is_same_v<std::decay_t<decltype(*std::get<0>(packed)[0])>, util::array<float, 2>>);

    EXPECT_EQ(reinterpret_cast<long>(std::get<1>(packed)[0]), 2048);
    EXPECT_EQ(reinterpret_cast<long>(std::get<1>(packed)[1]), 80);
    static_assert(std::is_same_v<std::decay_t<decltype(*std::get<1>(packed)[0])>, char>);

    EXPECT_EQ(reinterpret_cast<long>(std::get<2>(packed)[0]), 4096);
    EXPECT_EQ(reinterpret_cast<long>(std::get<2>(packed)[1]), 90);
    static_assert(std::is_same_v<std::decay_t<decltype(*std::get<2>(packed)[0])>, util::array<float, 4>>);

    EXPECT_EQ(reinterpret_cast<long>(std::get<3>(packed)[0]), 8192);
    EXPECT_EQ(reinterpret_cast<long>(std::get<3>(packed)[1]), 250);
    static_assert(std::is_same_v<std::decay_t<decltype(*std::get<3>(packed)[0])>, util::array<float, 1>>);
}

TEST(PackBuffers, packBufferPtrsA8)
{
    constexpr int Alignment = 8;
    char* packedBufferBase  = 0; // NOLINT

    size_t bufferSizes = 5;
    auto* p1           = reinterpret_cast<double*>(1024);
    auto* p2           = reinterpret_cast<uint8_t*>(2048);
    auto* p3           = reinterpret_cast<util::array<int, 4>*>(4096);
    auto* p4           = reinterpret_cast<int*>(8192);

    auto packed = packBufferPtrs<Alignment>(packedBufferBase, bufferSizes, p1, p2, p3, p4);

    EXPECT_EQ(reinterpret_cast<long>(std::get<0>(packed)[0]), 1024);
    EXPECT_EQ(reinterpret_cast<long>(std::get<0>(packed)[1]), 0);
    static_assert(std::is_same_v<std::decay_t<decltype(*std::get<0>(packed)[0])>, util::array<float, 2>>);

    EXPECT_EQ(reinterpret_cast<long>(std::get<1>(packed)[0]), 2048);
    EXPECT_EQ(reinterpret_cast<long>(std::get<1>(packed)[1]), 40);
    static_assert(std::is_same_v<std::decay_t<decltype(*std::get<1>(packed)[0])>, uint8_t>);

    EXPECT_EQ(reinterpret_cast<long>(std::get<2>(packed)[0]), 4096);
    EXPECT_EQ(reinterpret_cast<long>(std::get<2>(packed)[1]), 48); // round up from 45
    static_assert(std::is_same_v<std::decay_t<decltype(*std::get<2>(packed)[0])>, util::array<float, 4>>);

    EXPECT_EQ(reinterpret_cast<long>(std::get<3>(packed)[0]), 8192);
    EXPECT_EQ(reinterpret_cast<long>(std::get<3>(packed)[1]), 128);
    static_assert(std::is_same_v<std::decay_t<decltype(*std::get<3>(packed)[0])>, util::array<float, 1>>);
}

TEST(PackBuffers, computeByteOffsetsDynamic)
{
    std::vector<size_t> numElements{3, 5, 2};
    auto byteOffsets = computeByteOffsets(numElements, 4, 8);
    std::vector<size_t> ref{0, 16, 40, 48};
    EXPECT_EQ(byteOffsets, ref);
}

TEST(PackBuffers, packAllocBuffer)
{
    std::vector<char> scratch;
    std::vector<size_t> numElements{3, 5, 2};
    auto offsets = packAllocBuffer<int>(scratch, numElements, 8);
    EXPECT_EQ(offsets.size(), 3);

    std::vector<int*> ref(4, (int*)scratch.data());
    int* base = reinterpret_cast<int*>(scratch.data());
    EXPECT_EQ(base, offsets[0]);
    EXPECT_EQ(base + 4, offsets[1]);
    EXPECT_EQ(base + 10, offsets[2]);
}
