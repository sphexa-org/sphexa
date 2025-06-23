/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  General purpose utilities
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

namespace util
{

namespace detail
{

template<std::size_t... Is, class F, class... Tuples>
inline constexpr auto tupleMapImpl(std::index_sequence<Is...>, F&& f, Tuples&&... tuples)
{
    return std::make_tuple(
        [&f](auto i, auto&&... ts) -> decltype(auto) {
            return f(std::get<i>(std::forward<decltype(ts)>(ts))...);
        }(std::integral_constant<std::size_t, Is>(), std::forward<Tuples>(tuples)...)...);
}

} // namespace detail

template<class F, class Tuple, class... Tuples>
inline constexpr auto tupleMap(F&& f, Tuple&& tuple, Tuples&&... tuples)
{
    constexpr auto n = std::tuple_size_v<std::decay_t<Tuple>>;
    static_assert((... && (std::tuple_size_v<std::decay_t<Tuples>> == n)), "All tuples must have same size");
    return detail::tupleMapImpl(std::make_index_sequence<n>(), std::forward<F>(f), std::forward<Tuple>(tuple),
                                std::forward<Tuples>(tuples)...);
}

//! @brief Utility to call function with each element in tuples
template<class F, class... Tuples>
inline constexpr void for_each_tuple(F&& f, Tuples&&... tuples)
{
    tupleMap(
        [&f](auto&&... args)
        {
            f(std::forward<decltype(args)>(args)...);
            return 0;
        },
        std::forward<Tuples>(tuples)...);
}

//! @brief convert an index_sequence into a tuple of integral constants (e.g. for use with for_each_tuple)
template<std::size_t... Is>
constexpr auto makeIntegralTuple(std::index_sequence<Is...>)
{
    return std::make_tuple(std::integral_constant<size_t, Is>{}...);
}

//! @brief Select tuple elements specified by the argument sequence
template<class Tuple, std::size_t... Ints>
std::tuple<std::tuple_element_t<Ints, std::decay_t<Tuple>>...> selectTuple(Tuple&& tuple, std::index_sequence<Ints...>)
{
    return {std::get<Ints>(std::forward<Tuple>(tuple))...};
}

template<std::size_t... Is>
constexpr auto
indexSequenceReverse(std::index_sequence<Is...> const&) -> decltype(std::index_sequence<sizeof...(Is) - 1U - Is...>{});

template<std::size_t N>
using makeIndexSequenceReverse = decltype(indexSequenceReverse(std::make_index_sequence<N>{}));

//! @brief Create a new tuple by reversing the element order of the argument tuple
template<class Tuple>
decltype(auto) reverse(Tuple&& tuple)
{
    return selectTuple(std::forward<Tuple>(tuple), makeIndexSequenceReverse<std::tuple_size_v<std::decay_t<Tuple>>>{});
}

//! @brief Return a new tuple without the last element of the argument tuple
template<class Tp>
constexpr auto discardLastElement(Tp&& tp)
{
    return selectTuple(std::forward<Tp>(tp), std::make_index_sequence<std::tuple_size_v<std::decay_t<Tp>> - 1>{});
}

} // namespace util
