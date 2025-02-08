//
// Created by Noah Kubli on 07.02.2025.
//

#pragma once

#include <array>

#include "cstone/util/array.hpp"

namespace disk
{
template<typename T>
struct is_array : std::false_type
{
};

template<typename T, size_t N>
struct is_array<util::array<T, N>> : std::true_type
{
};

template<typename T, size_t N>
struct is_array<std::array<T, N>> : std::true_type
{
};

template<typename T>
inline constexpr bool is_array_v = is_array<T>::value;

template<typename T>
concept ArrayType = is_array_v<T>;

template<typename T>
struct value_type : std::type_identity<T>
{
};

template<typename T>
requires ArrayType<T> struct value_type<T> : std::type_identity<typename T::value_type>
{
};

template<typename T>
using value_type_t = value_type<T>::type;

template<typename T>
concept arithmetic = std::is_arithmetic_v<T>;

template<typename... Ts>
struct is_arithmetic_or_arrays
{
    inline static constexpr bool value = ((arithmetic<Ts> || ArrayType<Ts>)&&...);
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
struct BufferSize : std::integral_constant<size_t, 0>
{
};

template<typename T, typename... Ts>
struct BufferSize<T, Ts...> : std::integral_constant<size_t, 1 + BufferSize<Ts...>::value>
{
};

template<ArrayType T, typename... Ts>
struct BufferSize<T, Ts...> : std::integral_constant<size_t, std::tuple_size_v<T> + BufferSize<Ts...>::value>
{
};

template<typename... T>
struct BufferSize<std::tuple<T...>> : std::integral_constant<size_t, BufferSize<std::decay_t<T>...>::value>
{
};

template<typename T>
inline constexpr size_t buffer_size_v = BufferSize<std::decay_t<T>>::value;

} // namespace disk