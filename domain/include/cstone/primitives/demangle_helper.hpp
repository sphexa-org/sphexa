/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

#pragma once

#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>

// gcc and clang both provide this header
#if __has_include(<cxxabi.h>)
#include <cxxabi.h>
namespace cstone::debug::detail
{
using support_cxxabi    = std::true_type;
constexpr auto demangle = abi::__cxa_demangle;
} // namespace cstone::debug::detail
#else
namespace cstone::debug::detail
{
using support_cxxabi = std::false_type;
template<typename... Ts>
constexpr char* demangle(Ts... ts)
{
    return nullptr;
}
} // namespace cstone::debug::detail
#endif

// --------------------------------------------------------------------
namespace cstone::debug::detail
{
// default : use built-in typeid to get the best info we can
template<typename T, typename Enabled = std::false_type>
struct demangle_helper
{
    char const* type_id() const { return typeid(T).name(); }
};

// if available : demangle an arbitrary c++ type using gnu utility
template<typename T>
struct demangle_helper<T, std::true_type>
{
    demangle_helper()
        : demangled_{demangle(typeid(T).name(), nullptr, nullptr, nullptr), std::free}
    {
    }

    char const* type_id() const { return demangled_ ? demangled_.get() : typeid(T).name(); }

private:
    // would prefer decltype(&std::free) here but clang overloads it for host/device code
    std::unique_ptr<char, void (*)(void*)> demangled_;
};

template<typename T>
using cxx_type_id = demangle_helper<T, support_cxxabi>;
} // namespace cstone::debug::detail

// --------------------------------------------------------------------
// print type information
// usage : std::cout << debug::print_type<args...>("separator")
// separator is appended if the number of types > 1
// --------------------------------------------------------------------
namespace cstone::debug
{
template<typename T = void> // print a single type
inline std::string print_type(char const* = "")
{
    return std::string(detail::cxx_type_id<T>().type_id());
}

template<> // fallback for an empty type
inline std::string print_type<>(char const*)
{
    return "<>";
}

template<typename T, typename... Args> // print a list of types
inline std::enable_if_t<sizeof...(Args) != 0, std::string> print_type(char const* delim = "")
{
    std::string temp(print_type<T>());
    return temp + delim + print_type<Args...>(delim);
}
} // namespace cstone::debug
