//
// Created by Noah Kubli on 07.02.2025.
//

#pragma once

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

#include <mpi.h>

#include "buffer_reduce_concepts.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/util/array.hpp"

namespace disk
{
namespace buffer
{
template<typename Fn, typename Tuple, array_type buffer_type>
requires(std::tuple_size_v<std::decay_t<buffer_type>> ==
         flattened_size_v<Tuple>) void for_each_buffer(Fn&& f, buffer_type&& buffer, Tuple&& args_tuple)
{
    size_t i_buffer      = 0;
    auto   access_buffer = [&i_buffer, &f, &buffer](auto&& arg)
    {
        if constexpr (array_type<std::decay_t<decltype(arg)>>)
        {
            for (size_t i_array = 0; i_array < arg.size(); i_array++)
            {
                f(buffer[i_buffer], arg[i_array]);
                i_buffer++;
            }
        }
        else
        {
            f(buffer[i_buffer], arg);
            i_buffer++;
        }
    };
    for_each_tuple([&](auto&& res) { access_buffer(std::forward<decltype(res)>(res)); }, args_tuple);
}

template<bufferable_types... T>
auto makeBuffer(const T&... args)
{
    using value_type             = std::common_type_t<value_type_t<std::decay_t<T>>...>;
    constexpr size_t buffer_size = flattened_size_v<std::tuple<T...>>;

    std::array<value_type, buffer_size> buffer;
    for_each_buffer([](auto& buf_element, const auto& value) { buf_element = value; }, buffer, std::tie(args...));
    return buffer;
}

template<bufferable_types... Ts, typename T, size_t N>
requires(N == flattened_size_v<std::tuple<Ts...>>) auto extractBuffer(const std::array<T, N>& buffer)
{
    using RetTuple = std::tuple<std::remove_reference_t<Ts>...>;
    if constexpr (sizeof...(Ts) == 1)
    {
        std::tuple_element_t<0, RetTuple> result;
        for_each_buffer([](const auto& buf_element, auto& res) { res = buf_element; }, buffer, std::tie(result));
        return result;
    }
    else
    {
        RetTuple result;
        for_each_buffer([](const auto& buf_element, auto& res) { res = buf_element; }, buffer, result);
        return result;
    }
}

//! @brief Copy arguments of the same arithmetic type and of array of this type into a buffer,
//! to collect multiple MPI calls into one; returns a tuple if there is more than one argument.
template<bufferable_types... T>
auto mpiAllreduceSum(const T&... args)
{
    auto buffer = makeBuffer(args...);

    MPI_Allreduce(MPI_IN_PLACE, buffer.data(), buffer.size(), MpiType<value_type_t<decltype(buffer)>>{}, MPI_SUM,
                  MPI_COMM_WORLD);

    return extractBuffer<T...>(buffer);
}
} // namespace buffer
} // namespace disk
