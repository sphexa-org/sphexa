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

#if defined(__CUDACC__) and (CUDART_VERSION < 12040)
#include <thrust/tuple.h>
#elif defined(__CUDACC__)
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
    return impl::get<N>(std::forward<T>(tup));
}

template<class... Ts>
HOST_DEVICE_FUN constexpr tuple<Ts&...> tie(Ts&... args) noexcept
{
    return impl::tuple<Ts&...>(args...);
}

} // namespace util

// Thrust tuples in CUDA are now cuda::std tuples for which structured bindings have been added in CUDA 12.4
#if defined(__CUDACC__) and (CUDART_VERSION < 12040)
namespace std
{
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
} // namespace std
#endif

