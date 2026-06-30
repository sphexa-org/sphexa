/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Functionality for temporary storage allocation in the supercluster neighborhood
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "cstone/execution.hpp"
#include "cstone/traversal/ijloop/common.hpp"
#include "cstone/util/type_list.hpp"

namespace cstone::ijloop
{

template<class... Ts>
consteval std::array<int, sizeof...(Ts)> typeSizes(std::tuple<Ts...>)
{
    return {sizeof(Ts)...};
}

template<std::size_t N, std::size_t M>
consteval std::array<int, N> mapSizes(std::array<int, N> const& resultSizes, std::array<int, M> outputSizes)
{
    std::array<int, N> indexMap;

    for (std::size_t i = 0; i < N; ++i)
    {
        indexMap[i] = -1;
        for (std::size_t j = 0; j < M; ++j)
        {
            if (outputSizes[j] == resultSizes[i])
            {
                indexMap[i]    = j;
                outputSizes[j] = -1;
                break;
            }
        }
    }

    return indexMap;
}

template<class Result, class Output>
consteval auto mapTemporarySizes(Result, Output)
{
    constexpr auto indexMap = mapSizes(typeSizes(Result{}), typeSizes(Output{}));
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return std::make_tuple(std::integral_constant<int, std::get<Is>(indexMap)>()...);
    }(std::make_index_sequence<indexMap.size()>());
}

template<class IndexMap, class Result, class Output>
auto allocateOrMapTemporaries(const execution::Gpu exec,
                              const LocalIndex firstBody,
                              const LocalIndex lastBody,
                              IndexMap,
                              Result,
                              const Output& output)
{
    return util::tupleMap(
        [&]<class Index, class T>(Index, T)
        {
            if constexpr (Index::value < 0)
            {
                auto holder = util::deviceAlloc<T[]>(exec, lastBody - firstBody);
                return std::make_tuple(holder.get() - firstBody, std::move(holder));
            }
            else { return std::make_tuple(std::get<Index::value>(output), nullptr); }
        },
        IndexMap{}, Result{});
}

/*! allocate or map temporary storage for output arrays required by the interaction kernel
 *
 * This function determines the required temporary storage for the given interaction kernel and either
 * allocates new device memory or maps to the provided output pointers, depending on whether the
 * temporary storage matches the output types. It returns a tuple containing the pointers to the
 * temporaries (or mapped outputs) and holders for any allocated memory.
 *
 * @param firstBody    index of the first body to process
 * @param lastBody     index of the last body to process
 * @param input        input data for the interaction
 * @param output       tuple of output pointers
 * @param interaction  interaction kernel (callable)
 * @return std::tuple of a tuple of temporary pointers and a data holder, which releases all allocated data as soon as
 * it is destructed
 */
template<class Config, class Tc, class ThP, class Input, class... Out, class Interaction>
auto allocateTemporaries(execution::Gpu exec,
                         LocalIndex firstBody,
                         LocalIndex lastBody,
                         Input const&,
                         std::tuple<Out*...> const& output,
                         Interaction&& interaction)
{
    if constexpr (Config::symmetric)
    {
        // in the symmetric case, temporary arrays are required iff the result of the interaction invocation returns
        // more values or data types of different sizes than the final output of the postamble
        using ParticleData =
            decltype(loadParticleData(std::declval<Tc*>(), std::declval<Tc*>(), std::declval<Tc*>(),
                                      std::declval<ThP>(), std::declval<Input>(), std::declval<LocalIndex>()));
        using Result = decltype(unwrapModifiers(std::forward<Interaction>(interaction)(
            std::declval<ParticleData>(), std::declval<ParticleData>(), std::declval<Vec3<Tc>>(), std::declval<Tc>())));

        constexpr auto ptrMap = mapTemporarySizes(Result{}, std::tuple<Out...>());

        auto ptrsAndHolders = allocateOrMapTemporaries(exec, firstBody, lastBody, ptrMap, Result{}, output);

        auto ptrs = util::tupleMap([](auto const& alloc) { return std::get<0>(alloc); }, ptrsAndHolders);
        auto holders =
            util::tupleMap([](auto&& alloc) { return std::get<1>(std::move(alloc)); }, std::move(ptrsAndHolders));

        return std::make_tuple(std::move(ptrs), std::move(holders));
    }
    else
    {
        // in the asymmetric case, no temporary storage is required ever
        return std::make_tuple(output, std::tuple());
    }
}
} // namespace cstone::ijloop
