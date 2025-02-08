//
// Created by Noah Kubli on 07.02.2025.
//

#pragma once

#include <array>
#include <tuple>
#include <type_traits>
#include <utility>

#include <mpi.h>

#include "buffer_reduce_helper.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/util/array.hpp"

namespace disk
{

template<typename Op, typename Tb, typename Tr>
requires(std::tuple_size_v<std::decay_t<Tb>> == buffer_size_v<Tr>) void for_each_buffer(Op&& op, Tb&& buffer,
                                                                                        Tr&& result)
{
    size_t i_buffer      = 0;
    auto   access_buffer = [&](auto&& arg)
    {
        if constexpr (is_array_v<std::decay_t<decltype(arg)>>)
        {
            for (size_t i = 0; i < arg.size(); i++)
            {
                op(buffer[i_buffer], arg[i]);
                i_buffer++;
            }
        }
        else
        {
            op(buffer[i_buffer], arg);
            i_buffer++;
        }
    };
    for_each_tuple([&](auto&& res) { access_buffer(std::forward<decltype(res)>(res)); }, result);
}

template<arithmetic T, size_t N>
void reduce_array(std::array<T, N>& buffer, int rank)
{
    MPI_Allreduce(MPI_IN_PLACE, buffer.data(), N, MpiType<T>{}, MPI_SUM, MPI_COMM_WORLD);
}

template<arithmetic_or_arrays... T>
requires same_value_types<T...> auto buffered_reduce(int rank, T&&... args)
{
    using value_type            = std::common_type_t<value_type_t<std::decay_t<T>>...>;
    constexpr size_t          N = buffer_size_v<std::tuple<T...>>;
    std::array<value_type, N> buffer;

    for_each_buffer([](auto& buf, const auto& arg) { buf = arg; }, buffer, std::tie(std::forward<T>(args)...));

    reduce_array(buffer, rank);

    std::tuple<std::decay_t<T>...> result;
    for_each_buffer([](const auto& buf, auto& res) { res = buf; }, buffer, result);
    return result;
}
} // namespace disk
