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

template<typename Fn, typename Tval, size_t N, typename Tuple>
requires(N == flattened_size_v<Tuple>) void for_each_buffer(Fn&& f, std::array<Tval, N>& buffer, Tuple&& args_tuple)
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

//! @brief Copy arguments of arithmetic type and of array of this type into a buffer;
//! to collect multiple MPI calls into one.
template<arithmetic_or_arrays... T>
requires same_value_types<T...> auto buffered_mpi_allreduce_sum(const T&... args)
{
    using value_type             = std::common_type_t<value_type_t<std::decay_t<T>>...>;
    constexpr size_t buffer_size = flattened_size_v<std::tuple<T...>>;

    std::array<value_type, buffer_size> buffer;

    for_each_buffer([](auto& buf_element, const auto& value) { buf_element = value; }, buffer, std::tie(args...));

    MPI_Allreduce(MPI_IN_PLACE, buffer.data(), buffer_size, MpiType<value_type>{}, MPI_SUM, MPI_COMM_WORLD);

    std::tuple<std::remove_reference_t<T>...> result;
    for_each_buffer([](const auto& buf_element, auto& res) { res = buf_element; }, buffer, result);
    return result;
}
} // namespace disk
