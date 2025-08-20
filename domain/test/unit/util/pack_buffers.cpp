/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
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
    long* p3  = nullptr;

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
    EXPECT_EQ(base, offsets[0].data());
    EXPECT_EQ(base + 4, offsets[1].data());
    EXPECT_EQ(base + 10, offsets[2].data());
}

TEST(PackBuffers, packAllocBufferPoly)
{
    std::vector<char> backing;

    int alignment = 8;
    std::array<std::size_t, 3> numElements{3, 7, 2};

    auto packed = packAllocBuffer(backing, TypeList<int, uint8_t, double>{}, numElements, alignment);

    std::size_t expectedSize = cstone::round_up(numElements[0] * sizeof(int), alignment) +
                               cstone::round_up(numElements[1] * sizeof(uint8_t), alignment) +
                               cstone::round_up(numElements[2] * sizeof(double), alignment);

    EXPECT_EQ(backing.size(), expectedSize);

    auto p0 = std::get<0>(packed);
    EXPECT_EQ(p0.size(), numElements[0]);
    EXPECT_EQ(reinterpret_cast<char*>(p0.data()) - backing.data(), 0);

    auto p1 = std::get<1>(packed);
    EXPECT_EQ(p1.size(), numElements[1]);
    EXPECT_EQ(reinterpret_cast<char*>(p1.data()) - backing.data(), 16);

    auto p2 = std::get<2>(packed);
    EXPECT_EQ(p2.size(), numElements[2]);
    EXPECT_EQ(reinterpret_cast<char*>(p2.data()) - backing.data(), 16 + 8);
}