/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Test common SFC functionality
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"
#include "cstone/sfc/common.hpp"

using namespace cstone;

template<class T>
void normalization32()
{
    EXPECT_EQ(toNBitInt<unsigned>(T(0)), 0);
    EXPECT_EQ(toNBitInt<unsigned>(T(0.0000001)), 0);
    EXPECT_EQ(toNBitInt<unsigned>(T(1)), 1023);

    EXPECT_EQ(toNBitInt<unsigned>(T(0.00489)), 5);
    EXPECT_EQ(toNBitInt<unsigned>(T(0.00196)), 2);
    EXPECT_EQ(toNBitInt<unsigned>(T(0.00391)), 4);
}

TEST(SfcCode, normalization32)
{
    normalization32<float>();
    normalization32<double>();
}

template<class T>
void normalization64()
{
    EXPECT_EQ(toNBitInt<uint64_t>(T(0)), 0);
    EXPECT_EQ(toNBitInt<unsigned>(T(0.0000001)), 0);
    EXPECT_EQ(toNBitInt<uint64_t>(T(1)), 2097151);

    EXPECT_EQ(toNBitInt<uint64_t>(T(0.00489)), 10255);
    EXPECT_EQ(toNBitInt<uint64_t>(T(0.00196)), 4110);
    EXPECT_EQ(toNBitInt<uint64_t>(T(0.00391)), 8199);
}

TEST(SfcCode, normalization64)
{
    normalization64<float>();
    normalization64<double>();
}

template<class T>
void normalizationCeil32()
{
    EXPECT_EQ(toNBitIntCeil<unsigned>(T(0)), 0);
    EXPECT_EQ(toNBitIntCeil<unsigned>(T(0.0000001)), 1);
    EXPECT_EQ(toNBitIntCeil<unsigned>(T(1)), 1023);

    EXPECT_EQ(toNBitIntCeil<unsigned>(T(0.00489)), 6);
    EXPECT_EQ(toNBitIntCeil<unsigned>(T(0.00196)), 3);
    EXPECT_EQ(toNBitIntCeil<unsigned>(T(0.00391)), 5);
}

TEST(SfcCode, normalizationCeil32)
{
    normalizationCeil32<float>();
    normalizationCeil32<double>();
}

template<class T>
void normalizationCeil64()
{
    EXPECT_EQ(toNBitIntCeil<uint64_t>(T(0)), 0);
    EXPECT_EQ(toNBitIntCeil<unsigned>(T(0.0000001)), 1);
    EXPECT_EQ(toNBitIntCeil<uint64_t>(T(1)), 2097151);

    EXPECT_EQ(toNBitIntCeil<uint64_t>(T(0.00489)), 10256);
    EXPECT_EQ(toNBitIntCeil<uint64_t>(T(0.00196)), 4111);
    EXPECT_EQ(toNBitIntCeil<uint64_t>(T(0.00391)), 8200);
}

TEST(SfcCode, normalizationCeil64)
{
    normalizationCeil64<float>();
    normalizationCeil64<double>();
}

TEST(SfcCode, zeroLowBits32)
{
    EXPECT_EQ((0b00111000u << 24u), zeroLowBits((0b00111111u << 24u), 3));
    EXPECT_EQ((0b00110000u << 24u), zeroLowBits((0b00111111u << 24u), 2));
    EXPECT_EQ((0b00100000u << 24u), zeroLowBits((0b00111111u << 24u), 1));
}

TEST(SfcCode, zeroLowBits64)
{
    EXPECT_EQ((0b0111000lu << 57u), zeroLowBits((0b0111111lu << 57u), 3));
    EXPECT_EQ((0b0110000lu << 57u), zeroLowBits((0b0111111lu << 57u), 2));
    EXPECT_EQ((0b0100000lu << 57u), zeroLowBits((0b0111111lu << 57u), 1));
}

TEST(SfcCode, log8ceil32)
{
    EXPECT_EQ(2, log8ceil(64u));
    EXPECT_EQ(3, log8ceil(100u));
}

TEST(SfcCode, log8ceil64)
{
    EXPECT_EQ(2, log8ceil(64lu));
    EXPECT_EQ(3, log8ceil(100lu));
}

TEST(SfcCode, treeLevel32)
{
    using CodeType = unsigned;
    EXPECT_EQ(0, treeLevel<CodeType>(1u << 30u));
    EXPECT_EQ(1, treeLevel<CodeType>(1u << 27u));
    EXPECT_EQ(10, treeLevel<CodeType>(1));
}

TEST(SfcCode, treeLevel64)
{
    using CodeType = uint64_t;
    EXPECT_EQ(0, treeLevel<CodeType>(1ul << 63u));
    EXPECT_EQ(1, treeLevel<CodeType>(1ul << 60u));
    EXPECT_EQ(21, treeLevel<CodeType>(1));
}

TEST(SfcCode, encodePlaceholderBit32)
{
    EXPECT_EQ(1, encodePlaceholderBit(0u, 0));
    EXPECT_EQ(0b1000, encodePlaceholderBit(0u, 3));
    EXPECT_EQ(0b1010, encodePlaceholderBit(pad(0b010u, 3), 3));
    EXPECT_EQ(01635, encodePlaceholderBit(06350000000u, 9));
}

TEST(SfcCode, decodePrefixLength32)
{
    EXPECT_EQ(0, decodePrefixLength(1u));
    EXPECT_EQ(3, decodePrefixLength(0b1000u));
    EXPECT_EQ(3, decodePrefixLength(0b1010u));
    EXPECT_EQ(9, decodePrefixLength(01635u));
}

TEST(SfcCode, decodePlaceholderbit32)
{
    EXPECT_EQ(0, decodePlaceholderBit(1u));
    EXPECT_EQ(0, decodePlaceholderBit(0b1000u));
    EXPECT_EQ(pad(0b010u, 3), decodePlaceholderBit(0b1010u));
    EXPECT_EQ(06350000000, decodePlaceholderBit(01635u));
}

TEST(SfcCode, decodePlaceholderbit2K32)
{
    using KeyType = uint32_t;
    auto [k1, k2] = decodePlaceholderBit2K(0b1001u);
    EXPECT_EQ(k1, nodeRange<KeyType>(1));
    EXPECT_EQ(k2, 2 * nodeRange<KeyType>(1));
}

TEST(SfcCode, decodePlaceholderbit2K32_2)
{
    using KeyType = uint32_t;
    auto [k1, k2] = decodePlaceholderBit2K(0100u);
    EXPECT_EQ(k1, 0);
    EXPECT_EQ(k2, nodeRange<KeyType>(2));
}

TEST(SfcCode, encodePlaceholderBit64)
{
    EXPECT_EQ(1, encodePlaceholderBit(0lu, 0));
    EXPECT_EQ(0b1000, encodePlaceholderBit(0lu, 3));
    EXPECT_EQ(0b1010, encodePlaceholderBit(pad(0b010lu, 3), 3));
    EXPECT_EQ(01635, encodePlaceholderBit(0635000000000000000000ul, 9));
}

TEST(SfcCode, decodePrefixLength64)
{
    EXPECT_EQ(0, decodePrefixLength(1ul));
    EXPECT_EQ(3, decodePrefixLength(0b1000ul));
    EXPECT_EQ(3, decodePrefixLength(0b1010ul));
    EXPECT_EQ(9, decodePrefixLength(01635ul));
}

TEST(SfcCode, decodePlaceholderbit64)
{
    EXPECT_EQ(0, decodePlaceholderBit(1ul));
    EXPECT_EQ(0, decodePlaceholderBit(0b1000ul));
    EXPECT_EQ(pad(0b010ul, 3), decodePlaceholderBit(0b1010ul));
    EXPECT_EQ(0635000000000000000000ul, decodePlaceholderBit(01635ul));
}

TEST(SfcCode, maskKey)
{
    EXPECT_EQ(maskKey(0lu), 0lu);
    EXPECT_EQ(maskKey(nodeRange<uint64_t>(0)), nodeRange<uint64_t>(0));
    EXPECT_EQ(maskKey(1lu), nodeRange<uint64_t>(0) + 1);
    EXPECT_EQ(maskKey(nodeRange<uint64_t>(0) - 1), nodeRange<uint64_t>(0) - 1 + nodeRange<uint64_t>(0));
}

TEST(SfcCode, unmaskKey)
{
    EXPECT_EQ(unmaskKey(0lu), 0lu);
    EXPECT_EQ(unmaskKey(nodeRange<uint64_t>(0)), nodeRange<uint64_t>(0));
    EXPECT_EQ(1lu, unmaskKey(nodeRange<uint64_t>(0) + 1));
    EXPECT_EQ(nodeRange<uint64_t>(0) - 1, unmaskKey(nodeRange<uint64_t>(0) - 1 + nodeRange<uint64_t>(0)));
}

TEST(SfcCode, octalDigit32)
{
    EXPECT_EQ(1, octalDigit(010000000000u, 0));
    EXPECT_EQ(1, octalDigit(01234567012u, 1));
    EXPECT_EQ(2, octalDigit(01234567012u, 2));
    EXPECT_EQ(3, octalDigit(01234567012u, 3));
    EXPECT_EQ(4, octalDigit(01234567012u, 4));
    EXPECT_EQ(5, octalDigit(01234567012u, 5));
    EXPECT_EQ(6, octalDigit(01234567012u, 6));
    EXPECT_EQ(7, octalDigit(01234567012u, 7));
    EXPECT_EQ(0, octalDigit(01234567012u, 8));
    EXPECT_EQ(1, octalDigit(01234567012u, 9));
    EXPECT_EQ(2, octalDigit(01234567012u, 10));
}

TEST(SfcCode, octalDigit64)
{
    EXPECT_EQ(1, octalDigit(01000000000000000000000ul, 0));
    EXPECT_EQ(1, octalDigit(0123456701200000000000ul, 1));
    EXPECT_EQ(2, octalDigit(0123456701200000000000ul, 2));
    EXPECT_EQ(3, octalDigit(0123456701200000000000ul, 3));
    EXPECT_EQ(4, octalDigit(0123456701200000000000ul, 4));
    EXPECT_EQ(5, octalDigit(0123456701200000000000ul, 5));
    EXPECT_EQ(6, octalDigit(0123456701200000000000ul, 6));
    EXPECT_EQ(7, octalDigit(0123456701200000000000ul, 7));
    EXPECT_EQ(0, octalDigit(0123456701200000000000ul, 8));
    EXPECT_EQ(1, octalDigit(0123456701200000000000ul, 9));
    EXPECT_EQ(2, octalDigit(0123456701200000000000ul, 10));

    EXPECT_EQ(7, octalDigit(0123456701200000000007ul, 21));
}

TEST(SfcCode, isAncestor)
{
    unsigned b = 010001;
    EXPECT_TRUE(isAncestor(1u, b));
    EXPECT_TRUE(isAncestor(010u, b));
    EXPECT_TRUE(isAncestor(011u, b));
    EXPECT_TRUE(isAncestor(017u, b));
    EXPECT_TRUE(isAncestor(0100u, b));
    EXPECT_TRUE(isAncestor(0107u, b));
    EXPECT_TRUE(isAncestor(010000u, b));
    EXPECT_TRUE(isAncestor(010001u, b));
    EXPECT_TRUE(isAncestor(010007u, b));

    EXPECT_FALSE(isAncestor(0111u, b));
    EXPECT_FALSE(isAncestor(010017u, b));
    EXPECT_FALSE(isAncestor(0100007u, b));
}

TEST(SfcCode, makePrefix)
{
    EXPECT_EQ(makePrefix(0u), 1u);
    EXPECT_EQ(makePrefix(1u), 010000000001u);
    EXPECT_EQ(makePrefix(01000000000u), 011u);
}

TEST(SfcCode, digitWeight)
{
    EXPECT_EQ(digitWeight(0), 0);
    EXPECT_EQ(digitWeight(1), -1);
    EXPECT_EQ(digitWeight(2), -2);
    EXPECT_EQ(digitWeight(3), -3);
    EXPECT_EQ(digitWeight(4), 3);
    EXPECT_EQ(digitWeight(5), 2);
    EXPECT_EQ(digitWeight(6), 1);
    EXPECT_EQ(digitWeight(7), 0);
}

TEST(SfcCode, enclosingBoxTrim)
{
    std::size_t code      = 0x0FF0000000000001;
    std::size_t reference = 0x0FC0000000000000;
    EXPECT_EQ(reference, enclosingBoxCode(code, 3));

    unsigned code_u      = 0x07F00001;
    unsigned reference_u = 0x07E00000;
    EXPECT_EQ(reference_u, enclosingBoxCode(code_u, 3));
}

TEST(SfcCode, enclosingBoxMaxLevel32)
{
    using CodeType = unsigned;
    CodeType code  = 0x0FF00001;
    CodeType probe = enclosingBoxCode(code, maxTreeLevel<CodeType>{});
    EXPECT_EQ(probe, code);
}

TEST(SfcCode, enclosingBoxMaxLevel64)
{
    using CodeType = uint64_t;
    CodeType code  = 0x0FF0000000000001;
    CodeType probe = enclosingBoxCode(code, maxTreeLevel<CodeType>{});
    EXPECT_EQ(probe, code);
}

TEST(SfcCode, smallestCommonBoxEqualCode)
{
    using CodeType = unsigned;
    CodeType code  = 0;
    auto [k1, k2]  = smallestCommonBox(code, code);
    util::tuple<CodeType, CodeType> reference{code, code + 1};
    EXPECT_EQ(k1, code);
    EXPECT_EQ(k2, code + 1);
}

TEST(SfcCode, smallestCommonBoxL1)
{
    using CodeType = unsigned;
    CodeType code1 = 0b00001001u << 24u;
    CodeType code2 = 0b00001010u << 24u;
    auto [k1, k2]  = smallestCommonBox(code1, code2);
    EXPECT_EQ(k1, 0b00001000u << 24u);
    EXPECT_EQ(k2, 0b000010000u << 24u);
}

TEST(SfcCode, smallestCommonBoxL0_32)
{
    using CodeType = unsigned;
    CodeType code1 = 0b00000001u << 24u;
    CodeType code2 = 0b00001010u << 24u;
    auto [k1, k2]  = smallestCommonBox(code1, code2);
    EXPECT_EQ(k1, 0);
    EXPECT_EQ(k2, 0b01u << 30u);
}

TEST(SfcCode, smallestCommonBoxL0_64)
{
    using CodeType = uint64_t;
    CodeType code1 = 0b0000001lu << 57u;
    CodeType code2 = 0b0001010lu << 57u;
    auto [k1, k2]  = smallestCommonBox(code1, code2);
    EXPECT_EQ(k1, 0lu);
    EXPECT_EQ(k2, 1lu << 63u);
}

TEST(SfcCode, padUtility)
{
    EXPECT_EQ(pad(0b011u, 3), 0b00011 << 27);
    EXPECT_EQ(pad(0b011ul, 3), 0b0011ul << 60);
}

TEST(SfcCode, lastNzPlace)
{
    EXPECT_EQ(lastNzPlace(pad(1u, 3)), 1);
    EXPECT_EQ(lastNzPlace(nodeRange<unsigned>(0)), 0);

    EXPECT_EQ(lastNzPlace(pad(1ul, 3)), 1);
    EXPECT_EQ(lastNzPlace(nodeRange<uint64_t>(0)), 0);

    EXPECT_EQ(lastNzPlace(0u), maxTreeLevel<uint32_t>{});
    EXPECT_EQ(lastNzPlace(0ul), maxTreeLevel<uint64_t>{});
    EXPECT_EQ(lastNzPlace(1u), maxTreeLevel<uint32_t>{});
    EXPECT_EQ(lastNzPlace(1ul), maxTreeLevel<uint64_t>{});
    EXPECT_EQ(lastNzPlace(4u), maxTreeLevel<uint32_t>{});
    EXPECT_EQ(lastNzPlace(4ul), maxTreeLevel<uint64_t>{});
    EXPECT_EQ(lastNzPlace(8u), maxTreeLevel<uint32_t>{} - 1);
    EXPECT_EQ(lastNzPlace(8ul), maxTreeLevel<uint64_t>{} - 1);
}

template<class KeyType>
void spanSfcRange()
{
    using I = KeyType;
    {
        std::vector<I> reference{0,     01000, 02000, 03000, 04000, 05000, 06000, 07000, 07100, 07200,
                                 07300, 07400, 07500, 07600, 07700, 07710, 07720, 07730, 07740, 07750,
                                 07760, 07770, 07771, 07772, 07773, 07774, 07775, 07776};
        std::vector<I> probe(reference.size());

        EXPECT_EQ(spanSfcRange(I(0), I(07777), probe.data()), 28);
        EXPECT_EQ(reference, probe);
    }
    {
        std::vector<I> reference{1,   2,    3,    4,    5,    6,    7,    010,  020,  030,  040,  050,  060,
                                 070, 0100, 0200, 0300, 0400, 0500, 0600, 0700, 0710, 0720, 0730, 0740, 0741};
        std::vector<I> probe(reference.size());

        EXPECT_EQ(spanSfcRange(I(1), I(0742), probe.data()), 26);
        EXPECT_EQ(reference, probe);
    }
    {
        std::vector<I> reference{041305, 041306, 041307, 041310, 041320, 041330, 041340, 041350, 041360,
                                 041370, 041400, 041500, 041600, 041700, 042000, 043000, 044000, 045000};
        std::vector<I> probe(reference.size());

        EXPECT_EQ(spanSfcRange(I(041305), I(046000), probe.data()), 18);
        EXPECT_EQ(reference, probe);
    }

    EXPECT_EQ(spanSfcRange(I(040000), I(050000)), 1);
    EXPECT_EQ(spanSfcRange(I(040000), I(060000)), 2);

    {
        std::vector<I> reference{pad(I(1), 3), pad(I(2), 3), pad(I(3), 3), pad(I(4), 3),
                                 pad(I(5), 3), pad(I(6), 3), pad(I(7), 3)};
        std::vector<I> probe(reference.size());

        EXPECT_EQ(spanSfcRange(pad(I(01), 3), nodeRange<I>(0), probe.data()), 7);
        EXPECT_EQ(reference, probe);
    }

    EXPECT_EQ(0, spanSfcRange(I(0), I(0)));
    EXPECT_EQ(0, spanSfcRange(I(1), I(1)));
}

TEST(SfcCode, spanSfcRange32) { spanSfcRange<unsigned>(); }

TEST(SfcCode, spanSfcRange64) { spanSfcRange<uint64_t>(); }
