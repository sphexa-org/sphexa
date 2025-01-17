/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Tests for compile-time-string tuple getters
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/util/value_list.hpp"

using namespace util;

/*

 // see comment in source code

TEST(ConstexprString, OperatorEq)
{
    constexpr StructuralString a("id");
    constexpr StructuralString b("id");

    static_assert(a == b);

    StructuralString c("id");
    StructuralString d("id");
    EXPECT_EQ(c, d);

    StructuralString e("id2");
    EXPECT_NE(d, e);
}

TEST(ValueList, element)
{
    using TestList = require_gcc_12::ValueList<1, 4, 2, 5, 3>;

    static_assert(require_gcc_12::ValueListElement<1, TestList>::value == 4);
}

TEST(ValueList, find)
{
    using TestList = require_gcc_12::ValueList<1, 4, 2, 5, 3>;

    static_assert(require_gcc_12::FindIndex<1, TestList>{} == 0);
    static_assert(require_gcc_12::FindIndex<4, TestList>{} == 1);
    static_assert(require_gcc_12::FindIndex<8, TestList>{} == 5);
}
*/

TEST(ValueList, nameGet)
{
    auto tup         = std::make_tuple(0, "alpha", 3.14);
    using FieldNames = FieldList<"id", "description", "number">;

    EXPECT_EQ(FieldListSize<FieldNames>{}, 3);

    {
        auto i = vl_detail::FindIndex<StructuralString("id"), FieldNames>{};
        EXPECT_EQ(i, 0);
    }
    {
        auto i = vl_detail::FindIndex<StructuralString("description"), FieldNames>{};
        EXPECT_EQ(i, 1);
    }
    {
        auto i = vl_detail::FindIndex<StructuralString("number"), FieldNames>{};
        EXPECT_EQ(i, 2);
    }

    auto f_id          = get<"id", FieldNames>(tup);
    auto f_description = get<"description", FieldNames>(tup);
    auto f_number      = get<"number", FieldNames>(tup);

    EXPECT_EQ(f_id, 0);
    EXPECT_EQ(f_description, "alpha");
    EXPECT_EQ(f_number, 3.14);
}

TEST(ValueList, fieldListGet)
{
    using FieldNames = FieldList<"F0", "F1", "F2">;

    {
        std::vector<int> vi{1, 2, 3};
        std::vector<float> vf{1., 2., 3.};
        std::vector<double> vd{10., 20., 30.};

        auto get_02 = get<FieldList<"F0", "F2">, FieldNames>(std::tie(vi, vf, vd));

        EXPECT_EQ(get<0>(get_02).back(), 3);
        EXPECT_EQ(get<1>(get_02).back(), 30.);

        EXPECT_EQ(get<0>(get_02).data(), vi.data());
        EXPECT_EQ(get<1>(get_02).data(), vd.data());
    }

    {
        std::vector<int> vi{1, 2, 3};
        std::vector<float> vf{1., 2., 3.};
        std::vector<double> vd{10., 20., 30.};

        auto get_rval = get<FieldList<"F0", "F2">, FieldNames>(std::make_tuple(vi, vf, vd));
        EXPECT_EQ(get<1>(get_rval).back(), 30.);
        EXPECT_NE(get<1>(get_rval).data(), vd.data());
    }
}

namespace util
{
namespace vl_detail
{

template<class T, class IntSeq>
struct MakeFieldListHelper
{
};

template<class T, size_t... Is>
struct MakeFieldListHelper<T, std::integer_sequence<size_t, Is...>>
{
    // +1 to accomodate the '\0' character
    using type = util::FieldList<util::StructuralString<std::char_traits<char>::length(T::fieldNames[Is]) + 1>(
        T::fieldNames[Is])...>;
};

} // namespace vl_detail
} // namespace util

//! @brief Construct a FieldList type from any type with a constexpr array<const char*, N> fieldNames member
template<class T>
struct MakeFieldList
{
    inline static constexpr int numFields = T::fieldNames.size();
    using Fields = typename vl_detail::MakeFieldListHelper<T, std::make_index_sequence<numFields>>::type;
};

struct Testset
{
    inline static constexpr auto fieldNames = make_array(FieldList<"a", "b", "c">{});
};

TEST(ValueList, MakeFieldList)
{
    auto tup = std::make_tuple(0, "alpha", 1.0);
    using FL = MakeFieldList<Testset>::Fields;
    auto e_a = util::get<"a", FL>(tup);
    EXPECT_EQ(e_a, 0);
}
