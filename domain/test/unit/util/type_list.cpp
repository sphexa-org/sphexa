/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Utility tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"
#include "cstone/util/type_list.hpp"

using namespace util;

TEST(TypeList, FuseTwo)
{
    using TL1 = TypeList<float, int>;
    using TL2 = TypeList<double, unsigned>;

    using TL_fused = FuseTwo<TL1, TL2>;
    using TL_ref   = TypeList<float, int, double, unsigned>;

    constexpr bool match = std::is_same_v<TL_fused, TL_ref>;
    EXPECT_TRUE(match);
}

TEST(TypeList, Fuse)
{
    using TL1 = TypeList<float, int>;
    using TL2 = TypeList<double, unsigned>;
    using TL3 = TypeList<char, short>;

    using TL_fused = Fuse<TL1, TL2, TL3>;
    using TL_ref   = TypeList<float, int, double, unsigned, char, short>;

    constexpr bool match = std::is_same_v<TL_fused, TL_ref>;
    EXPECT_TRUE(match);
}

TEST(TypeList, Repeat)
{
    using TL1 = TypeList<float, int>;

    using TL_repeated = Repeat<TL1, 3>;
    using TL_ref      = TypeList<float, int, float, int, float, int>;

    constexpr bool match = std::is_same_v<TL_repeated, TL_ref>;
    EXPECT_TRUE(match);
}

TEST(TypeList, FindIndexTuple1)
{
    using TupleType = std::tuple<float>;

    constexpr int floatIndex = FindIndex<float, TupleType>{};

    constexpr int outOfRange = FindIndex<unsigned, TupleType>{};

    EXPECT_EQ(0, floatIndex);
    EXPECT_EQ(1, outOfRange);
}

TEST(TypeList, FindIndexTuple2)
{
    using TupleType = std::tuple<float, int>;

    constexpr int floatIndex = FindIndex<float, TupleType>{};
    constexpr int intIndex   = FindIndex<int, TupleType>{};

    constexpr int outOfRange = FindIndex<unsigned, TupleType>{};

    EXPECT_EQ(0, floatIndex);
    EXPECT_EQ(1, intIndex);
    EXPECT_EQ(2, outOfRange);
}

TEST(TypeList, FindIndexTypeList1)
{
    using ListType = TypeList<float>;

    constexpr int floatIndex = FindIndex<float, ListType>{};

    constexpr int outOfRange = FindIndex<unsigned, ListType>{};

    EXPECT_EQ(0, floatIndex);
    EXPECT_EQ(1, outOfRange);
}

TEST(TypeList, FindIndexTypeList2)
{
    using ListType = TypeList<float, int>;

    constexpr int floatIndex = FindIndex<float, ListType>{};
    constexpr int intIndex   = FindIndex<int, ListType>{};

    constexpr int outOfRange = FindIndex<unsigned, ListType>{};

    EXPECT_EQ(0, floatIndex);
    EXPECT_EQ(1, intIndex);
    EXPECT_EQ(2, outOfRange);
}

TEST(TypeList, Contains)
{
    using ListType = TypeList<float, int>;

    constexpr bool hasFloat = Contains<float, ListType>{};
    constexpr bool hasInt   = Contains<int, ListType>{};
    constexpr bool hasUint  = Contains<unsigned, ListType>{};

    EXPECT_TRUE(hasFloat);
    EXPECT_TRUE(hasInt);
    EXPECT_FALSE(hasUint);
}

TEST(TypeList, FindIndexTupleRepeated)
{
    using TupleType = std::tuple<float, float, int>;

    constexpr int floatIndex = FindIndex<float, TupleType>{};

    constexpr int intIndex = FindIndex<int, TupleType>{};

    constexpr int outOfRange = FindIndex<unsigned, TupleType>{};

    EXPECT_EQ(0, floatIndex);
    EXPECT_EQ(2, intIndex);
    EXPECT_EQ(3, outOfRange);
}

TEST(TypeList, FindIndexTypeListRepeated)
{
    using TupleType = TypeList<float, float, int>;

    constexpr int floatIndex = FindIndex<float, TupleType>{};

    constexpr int intIndex = FindIndex<int, TupleType>{};

    constexpr int outOfRange = FindIndex<unsigned, TupleType>{};

    EXPECT_EQ(0, floatIndex);
    EXPECT_EQ(2, intIndex);
    EXPECT_EQ(3, outOfRange);
}

TEST(TypeList, TypeListElementAccess)
{
    using TupleType = TypeList<float, double, int, unsigned>;

    static_assert(TypeListSize<TupleType>{} == 4);
    static_assert(std::is_same_v<TypeListElement_t<0, TupleType>, float>);
    static_assert(std::is_same_v<TypeListElement_t<1, TupleType>, double>);
    static_assert(std::is_same_v<TypeListElement_t<2, TupleType>, int>);
    static_assert(std::is_same_v<TypeListElement_t<3, TupleType>, unsigned>);
}

TEST(TypeList, SubsetIndices)
{
    using SubList  = TypeList<short, unsigned>;
    using BaseList = TypeList<float, double, short, int, unsigned>;

    [[maybe_unused]] auto indices = subsetIndices(SubList{}, BaseList{});
    static_assert(std::is_same_v<decltype(indices), std::index_sequence<2, 4>>);
}

TEST(TypeList, SubsetIndicesOutOfRange)
{
    using SubList  = TypeList<short, unsigned>;
    using BaseList = TypeList<float, double, short, int, unsigned>;

    [[maybe_unused]] auto outOfRange = subsetIndices(BaseList{}, SubList{});

    constexpr std::size_t length = TypeListSize<SubList>{};
    static_assert(std::is_same_v<decltype(outOfRange), std::index_sequence<length, length, 0, length, 1>>);
}
