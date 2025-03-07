/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! \file
 * \brief  implementation of a compile-time size array that can be used on the host and device
 *
 * \author Sebastian Keller <keller@cscs.ch>
 */

#pragma once

#include <cmath>
#include <utility>

#include "cstone/cuda/annotation.hpp"

namespace util
{

template<class T>
constexpr int determineAlignment(int n)
{
    if (sizeof(T) * n % 16 == 0) { return 16; }
    else if (sizeof(T) * n % 8 == 0) { return 8; }
    else { return alignof(T); }
}

/*! \brief std::array-like compile-time size array
 * \tparam T element type
 * \tparam N number of elements
 *
 * The implementation corresponds to a device-qualified std::array minus support for length 0
 * plus arithmetic operations.
 */
template<class T, std::size_t N>
struct alignas(determineAlignment<T>(N)) array
{
    typedef T value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef value_type* iterator;
    typedef const value_type* const_iterator;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    // public for aggregate type, as in std::array
    T data_[N];

    // No explicit construct/copy/destroy for aggregate type.

    HOST_DEVICE_FUN constexpr iterator begin() noexcept { return iterator(data()); }

    HOST_DEVICE_FUN constexpr const_iterator begin() const noexcept { return const_iterator(data()); }

    HOST_DEVICE_FUN constexpr iterator end() noexcept { return iterator(data() + N); }

    HOST_DEVICE_FUN constexpr const_iterator end() const noexcept { return const_iterator(data() + N); }

    HOST_DEVICE_FUN constexpr reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }

    HOST_DEVICE_FUN constexpr const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }

    HOST_DEVICE_FUN constexpr reverse_iterator rend() noexcept { return reverse_iterator(begin()); }

    HOST_DEVICE_FUN constexpr const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }

    HOST_DEVICE_FUN constexpr const_iterator cbegin() const noexcept { return const_iterator(data()); }

    HOST_DEVICE_FUN constexpr const_iterator cend() const noexcept { return const_iterator(data() + N); }

    HOST_DEVICE_FUN constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(end()); }

    HOST_DEVICE_FUN constexpr const_reverse_iterator crend() const noexcept { return const_reverse_iterator(begin()); }

    HOST_DEVICE_FUN constexpr size_type size() const noexcept { return N; }

    HOST_DEVICE_FUN constexpr size_type max_size() const noexcept { return N; }

    [[nodiscard]] HOST_DEVICE_FUN constexpr bool empty() const noexcept { return size() == 0; }

    // Element access.
    HOST_DEVICE_FUN constexpr reference operator[](size_type n) noexcept { return data_[n]; }

    HOST_DEVICE_FUN constexpr const_reference operator[](size_type n) const noexcept { return data_[n]; }

    HOST_DEVICE_FUN constexpr reference front() noexcept { return *begin(); }

    HOST_DEVICE_FUN constexpr const_reference front() const noexcept { return data_[0]; }

    HOST_DEVICE_FUN constexpr reference back() noexcept { return *(end() - 1); }

    HOST_DEVICE_FUN constexpr const_reference back() const noexcept { return data_[N - 1]; }

    HOST_DEVICE_FUN constexpr pointer data() noexcept { return data_; }

    HOST_DEVICE_FUN constexpr const_pointer data() const noexcept { return data_; }

    HOST_DEVICE_FUN constexpr array<T, N>& operator=(const value_type& rhs) noexcept
    {
        auto assignAToB = [](T /*a*/, T b) { return b; };
        assignImpl(data(), rhs, assignAToB, std::make_index_sequence<N>{});
        return *this;
    }

    HOST_DEVICE_FUN constexpr array<T, N>& operator+=(const array<T, N>& rhs) noexcept
    {
        auto add = [](T a, T b) { return a + b; };
        assignImpl(data(), rhs.data(), add, std::make_index_sequence<N>{});
        return *this;
    }

    HOST_DEVICE_FUN constexpr array<T, N>& operator-=(const array<T, N>& rhs) noexcept
    {
        auto minus = [](T a, T b) { return a - b; };
        assignImpl(data(), rhs.data(), minus, std::make_index_sequence<N>{});
        return *this;
    }

    HOST_DEVICE_FUN constexpr array<T, N>& operator*=(const value_type& rhs) noexcept
    {
        auto mult = [](T a, T b) { return a * b; };
        assignImpl(data(), rhs, mult, std::make_index_sequence<N>{});
        return *this;
    }

    HOST_DEVICE_FUN constexpr array<T, N>& operator/=(const value_type& rhs) noexcept
    {
        auto divide = [](T a, T b) { return a / b; };
        assignImpl(data(), rhs, divide, std::make_index_sequence<N>{});
        return *this;
    }

private:
    template<class F, std::size_t... Is>
    HOST_DEVICE_FUN constexpr static void assignImpl(T* a, const T* b, F&& f, std::index_sequence<Is...>) noexcept
    {
        [[maybe_unused]] std::initializer_list<int> list{(a[Is] = f(a[Is], b[Is]), 0)...};
    }

    template<class F, std::size_t... Is>
    HOST_DEVICE_FUN constexpr static void assignImpl(T* a, const T& b, F&& f, std::index_sequence<Is...>) noexcept
    {
        [[maybe_unused]] std::initializer_list<int> list{(a[Is] = f(a[Is], b), 0)...};
    }
};

template<std::size_t I, class T, std::size_t N>
HOST_DEVICE_FUN constexpr T& get(array<T, N>& a_)
{
    return a_[I];
}

template<std::size_t I, class T, std::size_t N>
HOST_DEVICE_FUN constexpr const T& get(const array<T, N>& a_)
{
    return a_[I];
}

template<std::size_t I, class T, std::size_t N>
HOST_DEVICE_FUN constexpr T&& get(array<T, N>&& a_)
{
    return std::move(a_[I]);
}

template<std::size_t I, class T, std::size_t N>
HOST_DEVICE_FUN constexpr const T&& get(const array<T, N>&& a_)
{
    return std::move(a_[I]);
}

template<class T, std::size_t N>
HOST_DEVICE_FUN constexpr array<T, N> operator+(const array<T, N>& a, const array<T, N>& b)
{
    auto ret = a;
    return ret += b;
}

namespace detail
{

template<class T, std::size_t... Is>
HOST_DEVICE_FUN constexpr array<T, sizeof...(Is)> negateImpl(const array<T, sizeof...(Is)>& a,
                                                             std::index_sequence<Is...>)
{
    return array<T, sizeof...(Is)>{-a[Is]...};
}

} // namespace detail

template<class T, std::size_t N>
HOST_DEVICE_FUN constexpr array<T, N> operator-(const array<T, N>& a)
{
    return detail::negateImpl(a, std::make_index_sequence<N>{});
}

template<class T, std::size_t N>
HOST_DEVICE_FUN constexpr array<T, N> operator-(const array<T, N>& a, const array<T, N>& b)
{
    auto ret = a;
    return ret -= b;
}

template<class S, class T, std::size_t N>
HOST_DEVICE_FUN constexpr array<T, N> operator*(const array<T, N>& a, const S& b)
{
    auto ret = a;
    return ret *= b;
}

template<class S, class T, std::size_t N>
HOST_DEVICE_FUN constexpr array<T, N> operator*(const S& a, const array<T, N>& b)
{
    auto ret = b;
    return ret *= a;
}

namespace detail
{

template<class T, std::size_t... Is>
HOST_DEVICE_FUN constexpr bool eqImpl(const T* a, const T* b, std::index_sequence<Is...>)
{
    return ((a[Is] == b[Is]) && ...);
}

template<class T, std::size_t... Is>
HOST_DEVICE_FUN constexpr T dotImpl(const T* a, const T* b, std::index_sequence<Is...>)
{
    return ((a[Is] * b[Is]) + ...);
}

template<class T, std::size_t... Is>
HOST_DEVICE_FUN constexpr array<T, sizeof...(Is)> absImpl(const T* a, std::index_sequence<Is...>)
{
    return {std::abs(a[Is])...};
}

template<class T, std::size_t... Is>
HOST_DEVICE_FUN constexpr array<T, sizeof...(Is)> maxImpl(const T* a, const T* b, std::index_sequence<Is...>)
{
    return {(a[Is] > b[Is] ? a[Is] : b[Is])...};
}

template<class T, std::size_t... Is>
HOST_DEVICE_FUN constexpr array<T, sizeof...(Is)> minImpl(const T* a, const T* b, std::index_sequence<Is...>)
{
    return {(a[Is] < b[Is] ? a[Is] : b[Is])...};
}

template<int N, int I = 0>
struct LexicographicalCompare
{
    template<class T, class F>
    HOST_DEVICE_FUN constexpr static auto loop(const T* lhs, const T* rhs, F&& compare)
    {
        if (compare(lhs[I], rhs[I])) { return true; }
        if (compare(rhs[I], lhs[I])) { return false; }
        return LexicographicalCompare<N, I + 1>::loop(lhs, rhs, compare);
    }
};

template<int N>
struct LexicographicalCompare<N, N>
{
    template<class T, class F>
    HOST_DEVICE_FUN constexpr static auto loop(const T*, const T*, F&&)
    {
        return false;
    }
};

} // namespace detail

template<class T, std::size_t N>
HOST_DEVICE_FUN constexpr bool operator==(const array<T, N>& a, const array<T, N>& b)
{
    return detail::eqImpl(a.data(), b.data(), std::make_index_sequence<N>{});
}

template<class T, std::size_t N>
HOST_DEVICE_FUN constexpr bool operator!=(const array<T, N>& a, const array<T, N>& b)
{
    return !(a == b);
}

template<class T, std::size_t N>
HOST_DEVICE_FUN constexpr bool operator<(const array<T, N>& a, const array<T, N>& b)
{
    auto less = [](T a, T b) { return a < b; };
    return detail::LexicographicalCompare<N>::loop(a.data(), b.data(), less);
}

template<class T, std::size_t N>
HOST_DEVICE_FUN constexpr bool operator>(const array<T, N>& a, const array<T, N>& b)
{
    auto greater = [](T a, T b) { return a > b; };
    return detail::LexicographicalCompare<N>::loop(a.data(), b.data(), greater);
}

template<class T, std::size_t N>
HOST_DEVICE_FUN constexpr T dot(const array<T, N>& a, const array<T, N>& b)
{
    return detail::dotImpl(a.data(), b.data(), std::make_index_sequence<N>{});
}

template<class T, std::size_t N>
HOST_DEVICE_FUN constexpr T norm2(const array<T, N>& a)
{
    return dot(a, a);
}

// template<class T, std::size_t N>
// HOST_DEVICE_FUN constexpr T norm(const array<T, N>& a)
//{
//     return std::sqrt(norm2(a));
// }

template<class T, std::size_t N>
HOST_DEVICE_FUN constexpr array<T, N> abs(const array<T, N>& a)
{
    return detail::absImpl(a.data(), std::make_index_sequence<N>{});
}

//! @brief element-wise min between two arrays
template<class T, std::size_t N>
HOST_DEVICE_FUN constexpr array<T, N> min(const array<T, N>& a, const array<T, N>& b)
{
    return detail::minImpl(a.data(), b.data(), std::make_index_sequence<N>{});
}

//! @brief element-wise max between two arrays
template<class T, std::size_t N>
HOST_DEVICE_FUN constexpr array<T, N> max(const array<T, N>& a, const array<T, N>& b)
{
    return detail::maxImpl(a.data(), b.data(), std::make_index_sequence<N>{});
}

//! @brief min reduction of a single array
template<class T, std::size_t N>
HOST_DEVICE_FUN constexpr T min(const array<T, N>& a)
{
    T ret = a[0];

    for (std::size_t i = 1; i < N; i++)
    {
        ret = ret < a[i] ? ret : a[i];
    }

    return ret;
}

//! @brief max reduction of a single array
template<class T, std::size_t N>
HOST_DEVICE_FUN constexpr T max(const array<T, N>& a)
{
    T ret = a[0];

    for (std::size_t i = 1; i < N; i++)
    {
        ret = ret > a[i] ? ret : a[i];
    }

    return ret;
}

template<class T>
HOST_DEVICE_FUN constexpr array<T, 3> cross(const array<T, 3>& a, const array<T, 3>& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

template<class T>
constexpr HOST_DEVICE_FUN array<T, 3> makeVec3(array<T, 4> v)
{
    return array<T, 3>{v[0], v[1], v[2]};
}

} // namespace util

//! @brief specializations of tuple traits in std:: namespace to make structured binding work with arrays
namespace std
{

template<size_t N, class T, size_t N2>
struct tuple_element<N, util::array<T, N2>>
{
    typedef T type;
};

template<class T, size_t N>
struct tuple_size<util::array<T, N>>
{
    static const size_t value = N;
};

} // namespace std
