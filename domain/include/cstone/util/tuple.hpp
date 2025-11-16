/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Wrapper around different types of tuples compatible with device code
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <tuple>
#include "cstone/cuda/annotation.hpp"

#if defined(__CUDACC__)
#include <cuda/std/tuple>
#endif

namespace util
{

#if defined(__CUDACC__) and (CUDART_VERSION < 12040)
namespace impl = thrust;
#elif defined(__CUDACC__)
namespace impl = cuda::std;
#else
namespace impl = std;
#endif

template<class... Ts>
using tuple = impl::tuple<Ts...>;

template<std::size_t N, class T>
HOST_DEVICE_FUN constexpr auto get(T&& tup) noexcept
{
    return impl::get<N>(tup);
}

template<class... Ts>
HOST_DEVICE_FUN constexpr tuple<Ts&...> tie(Ts&... args) noexcept
{
    return impl::tuple<Ts&...>(args...);
}

// Thrust tuples in CUDA are now cuda::std tuples for which structured bindings have been added in CUDA 12.4
#if defined(__CUDACC__) and (CUDART_VERSION < 12040)
template<size_t N, class... Ts>
struct tuple_element<N, thrust::tuple<Ts...>>
{
    typedef typename thrust::tuple_element<N, thrust::tuple<Ts...>>::type type;
};

template<class... Ts>
struct tuple_size<thrust::tuple<Ts...>>
{
    static const int value = thrust::tuple_size<thrust::tuple<Ts...>>::value;
};
#endif

} // namespace util

namespace util
{

template<class Tuple>
struct TuplePlusImpl
{
    template<std::size_t... Is>
    HOST_DEVICE_FUN Tuple operator()(const Tuple& a, const Tuple& b, std::index_sequence<Is...>)
    {
        return Tuple((util::get<Is>(a) + util::get<Is>(b))...);
    }
};

/*! @brief generic tuple addition functor that works for both thrust and std tuples
 *
 * @tparam Tuple   the kind of tuple to be added, e.g. thrust::tuple<int, double>
 */
template<class Tuple>
struct TuplePlus
{
    HOST_DEVICE_FUN Tuple operator()(const Tuple& a, const Tuple& b)
    {
        return TuplePlusImpl<Tuple>{}(a, b, std::make_index_sequence<std::tuple_size_v<Tuple>>{});
    }
};

} // namespace util