/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Constexpr string based on C++20 structural types
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <string_view>

namespace util
{

//! @brief constexpr string as structural type for use as non-type template parameter (C++20)
template<size_t N>
struct StructuralString
{
    //! @brief construction from string literal
    constexpr StructuralString(const char (&str)[N]) noexcept { std::copy_n(str, N, value); }

    //! @brief construction from string_view or const char*, needs explicit specification of template arg N
    constexpr StructuralString(std::string_view str) noexcept { std::copy_n(str.data(), N, value); }

    char value[N];
};

template<size_t N1, size_t N2>
constexpr bool operator==(const StructuralString<N1>& a, const StructuralString<N2>& b)
{
    return (N1 == N2) && std::equal(a.value, a.value + N1, b.value);
}

template<size_t N1, size_t N2>
constexpr StructuralString<N1 + N2 - 1> operator+(const StructuralString<N1>& a, const StructuralString<N2>& b)
{
    char value[N1 + N2 - 1];
    std::copy_n(a.value, N1 - 1, value);
    std::copy_n(b.value, N2, value + N1 - 1);
    return StructuralString(value);
}

} // namespace util
