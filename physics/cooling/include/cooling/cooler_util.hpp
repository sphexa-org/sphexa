//
// Created by Noah Kubli on 01.03.2024.
//

#pragma once

#include <functional>
#include <tuple>

namespace cooling
{

template<size_t... J>
struct is_all_equal
{
    using type = std::bool_constant<true>;
};

template<size_t I, size_t... J>
struct is_all_equal<I, J...>
{
    using type = std::bool_constant<((I == J) && ...)>;
};

template<typename... Tuples>
concept same_sizes = (is_all_equal<std::tuple_size_v<Tuples>...>::type::value);

//! @brief For Tuples A, B ... call f(a1, b1 ...), f(a2, b2 ...)
template<typename... Tuples, typename F>
requires same_sizes<std::decay_t<Tuples>...> void for_each_tuples(F&& f, Tuples&&... tuples)
{
    auto f_i = [&](auto I) { return f(std::get<I>(std::forward<Tuples>(tuples))...); };

    auto iterate_each = [&f_i]<size_t... Is>(std::index_sequence<Is...>)
    {
        (f_i(std::integral_constant<size_t, Is>{}), ...);
    };

    constexpr size_t n_elements = std::min({std::tuple_size_v<std::decay_t<Tuples>>...});
    iterate_each(std::make_index_sequence<n_elements>{});
}

} // namespace cooling
