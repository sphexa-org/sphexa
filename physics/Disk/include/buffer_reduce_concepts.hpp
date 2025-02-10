//
// Created by Noah Kubli on 07.02.2025.
//

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>

#include "cstone/util/array.hpp"

namespace disk
{

template<typename T>
concept array_type = requires(T t, std::size_t i)
{
    t[i];
    t.size();
    std::tuple_size_v<T>;
    typename T::value_type;
};

template<typename T>
struct value_type : std::type_identity<T>
{
};

template<array_type T>
struct value_type<T> : std::type_identity<typename T::value_type>
{
};

template<typename T>
using value_type_t = value_type<T>::type;

template<typename... Ts>
struct is_arithmetic_or_arrays
{
    inline static constexpr bool value = ((std::is_arithmetic_v<Ts> || array_type<Ts>)&&...);
};

template<typename... T>
concept arithmetic_or_arrays = is_arithmetic_or_arrays<std::decay_t<T>...>::value;

template<typename... T>
struct is_same_value_types : std::true_type
{
};

template<typename T1, typename... Ts>
struct is_same_value_types<T1, Ts...> : std::bool_constant<(std::is_same_v<value_type_t<T1>, value_type_t<Ts>> && ...)>
{
};

template<typename... T>
concept same_value_types = is_same_value_types<std::decay_t<T>...>::value;

template<typename... T>
struct flattened_size : std::integral_constant<std::size_t, 0>
{
};

template<typename T, typename... Ts>
struct flattened_size<T, Ts...> : std::integral_constant<std::size_t, 1 + flattened_size<Ts...>::value>
{
};

template<array_type T, typename... Ts>
struct flattened_size<T, Ts...>
    : std::integral_constant<std::size_t, std::tuple_size_v<T> + flattened_size<Ts...>::value>
{
};

template<typename... T>
struct flattened_size<std::tuple<T...>> : std::integral_constant<std::size_t, flattened_size<std::decay_t<T>...>::value>
{
};

template<typename T>
inline constexpr std::size_t flattened_size_v = flattened_size<std::decay_t<T>>::value;

} // namespace disk