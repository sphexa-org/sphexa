/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Utility functions to use compile-time strings as arguments to get on tuples
 *
 * Needs C++20 structural types
 */

#pragma once

#include "cstone/util/constexpr_string.hpp"
#include "cstone/util/tuple_util.hpp"

namespace require_gcc_12
{

/*! @brief Generic ValueList for arbitrary compile-time constant values
 *
 * Supersedes FieldList (restricted to StructuralString). Not supported by GCC 11 due to internal compiler errors.
 */
template<auto...>
struct ValueList
{
};

template<class VL>
struct ValueListSize
{
};

template<template<auto...> class VL, auto... Vs>
struct ValueListSize<VL<Vs...>> : public std::integral_constant<std::size_t, sizeof...(Vs)>
{
};

template<size_t I, class VL>
struct ValueListElement
{
};

//! @brief Element type retrieval: recursion, strip one element
template<size_t I, template<auto...> class VL, auto Head, auto... Tail>
struct ValueListElement<I, VL<Head, Tail...>> : public ValueListElement<I - 1, VL<Tail...>>
{
};

//! @brief Element type retrieval: endpoint, Head is the desired type
template<auto Head, template<auto...> class TL, auto... Tail>
struct ValueListElement<0, TL<Head, Tail...>>
{
    constexpr inline static auto value = Head;
};

//! @brief Element type retrieval: out of bounds detection
template<size_t I, template<auto...> class VL>
struct ValueListElement<I, VL<>>
{
    static_assert(I < ValueListSize<VL<>>{}, "ValueListElement access out of range");
};

namespace detail
{

//! @brief Recursion to check the next field N+1
template<int N, auto V, class VL, class Match = void>
struct MatchValue : std::integral_constant<size_t, MatchValue<N + 1, V, VL>{}>
{
};

//! @brief recursion stop when V == ValueListElement<N, VL>::value is true
template<int N, auto V, class VL>
struct MatchValue<N, V, VL, std::enable_if_t<V == ValueListElement<N, VL>::value>> : std::integral_constant<size_t, N>
{
};

} // namespace detail

/*! @brief Meta function to return the first index in VL whose value matches V
 *
 *  If there are more than one, the first occurrence will be returned.
 *  If there is no such type, the size of VL will be returned.
 */
template<auto V, class VL>
struct FindIndex
{
};

/*! @brief Specialization to only enable this trait if TL has non-type template parameters
 *
 * @tparam V          a value to look for in the template parameters of TL
 * @tparam VL         a non-type template template parameter, e.g. an instance of ValueList
 * @tparam Vs         template parameters of VL
 * @tparam Comparison comparison operation
 *
 *  Note that @p V is added to @pa VL as a sentinel to terminate the recursion
 *  and prevent an out of bounds tuple access compiler error.
 */
template<auto V, template<auto...> class VL, auto... Vs>
struct FindIndex<V, VL<Vs...>> : detail::MatchValue<0, V, VL<Vs..., V>>
{
};

} // namespace require_gcc_12

namespace util
{

/*! @brief Simplified ValueList restricted to StructuralString
 *
 * This can be removed and replaced by the more general ValueList once GCC 11 support is no longer needed.
 */
template<StructuralString... Fields>
struct FieldList
{
};

template<StructuralString... F1, StructuralString... F2>
constexpr auto operator+(FieldList<F1...>, FieldList<F2...>)
{
    return FieldList<F1..., F2...>{};
}

namespace detail
{
template<class VL1, class VL2>
struct FuseValueListImpl
{
};

template<StructuralString... F1, StructuralString... F2>
struct FuseValueListImpl<FieldList<F1...>, FieldList<F2...>>
{
    using type = FieldList<F1..., F2...>;
};
} // namespace detail

//! @brief Fuse (concatenate) two FieldLists
template<class VL1, class VL2>
using FuseValueList = typename detail::FuseValueListImpl<VL1, VL2>::type;

template<class VL>
struct FieldListSize
{
};

template<StructuralString... Vs>
struct FieldListSize<FieldList<Vs...>> : public std::integral_constant<std::size_t, sizeof...(Vs)>
{
};

//! @brief convert FieldList to std::tuple
template<StructuralString... Fields>
constexpr auto make_tuple(FieldList<Fields...>)
{
    return std::make_tuple(Fields...);
}

//! @brief convert FieldList to constexpr std::array
template<StructuralString... Fields>
constexpr auto make_array(FieldList<Fields...>)
{
    return std::array{Fields.value...};
}

namespace vl_detail
{

template<size_t I, bool match, StructuralString SS, StructuralString Head, StructuralString... Tail>
struct MatchField_helper : public MatchField_helper<I + 1, SS == Head, SS, Tail...>
{
};

template<size_t I, StructuralString SS, StructuralString Head, StructuralString... Tail>
struct MatchField_helper<I, true, SS, Head, Tail...> : std::integral_constant<size_t, I>
{
};

template<size_t I, StructuralString SS, class VL>
struct MatchField
{
};

//! @brief Element retrieval: recursion, strip one element
template<size_t I, StructuralString SS, StructuralString Head, StructuralString... Tail>
struct MatchField<I, SS, FieldList<Head, Tail...>> : public MatchField_helper<I, SS == Head, SS, Tail...>
{
};

template<StructuralString V, class VL>
struct FindIndex
{
};

template<StructuralString V, StructuralString... Vs>
struct FindIndex<V, FieldList<Vs...>> : MatchField<0, V, FieldList<Vs..., V>>
{
};

} // namespace vl_detail

//! @brief Access a field of a tuple with named fields
template<StructuralString F, class FieldNames, class Tuple>
decltype(auto) get(Tuple&& tuple)
{
    return get<vl_detail::FindIndex<F, FieldNames>{}>(std::forward<Tuple>(tuple));
}

//! @brief Access fields of a tuple with FieldLists
template<class FL, class FieldNames, class Tuple>
auto get(Tuple&& tuple)
{
    auto get_ = []<StructuralString... Fs>(FieldList<Fs...>, Tuple&& tuple) // NOLINT
    {
        using Seq = std::index_sequence<vl_detail::FindIndex<Fs, FieldNames>{}...>;
        return selectTuple(std::forward<Tuple>(tuple), Seq{});
    };

    return get_(FL{}, std::forward<Tuple>(tuple));
}

} // namespace util
