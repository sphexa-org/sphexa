/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Functionality for temporary storage allocation in the supercluster neighborhood
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include "cstone/traversal/ijloop/common.hpp"
#include "cstone/util/type_list.hpp"

namespace cstone::ijloop::gpu_supercluster_nb_list_neighborhood_detail
{

struct None
{
};

template<class T, class... Ts>
consteval auto tail(util::TypeList<T, Ts...>)
{
    return util::TypeList<Ts...>{};
}

consteval auto tail(util::TypeList<>) { return util::TypeList<>{}; }

template<class Size, class IndexedOutSizes>
consteval auto findMatchingOutput(Size, IndexedOutSizes)
{
    if constexpr (util::TypeListSize<IndexedOutSizes>::value == 0) { return None{}; }
    else
    {
        using Head      = util::TypeListElement_t<0, IndexedOutSizes>;
        using HeadIndex = util::TypeListElement_t<0, Head>;
        using HeadSize  = util::TypeListElement_t<1, Head>;
        if constexpr (std::is_same_v<Size, HeadSize>)
            return HeadIndex{};
        else
            return findMatchingOutput(Size{}, tail(IndexedOutSizes{}));
    }
}

template<class Index, class IndexedOutSizes>
consteval auto dropByIndex(Index, IndexedOutSizes)
{
    if constexpr (std::is_same_v<Index, None>) { return IndexedOutSizes{}; }
    else if constexpr (util::TypeListSize<IndexedOutSizes>::value == 0) { return util::TypeList<>{}; }
    else
    {
        using Head      = util::TypeListElement_t<0, IndexedOutSizes>;
        using HeadIndex = util::TypeListElement_t<0, Head>;
        if constexpr (std::is_same_v<Index, HeadIndex>)
            return tail(IndexedOutSizes{});
        else
            return util::FuseTwo<util::TypeList<Head>, decltype(dropByIndex(Index{}, tail(IndexedOutSizes{})))>{};
    }
}

template<class TmpSizes, class OutSizes>
consteval auto mapTemporarySizes(TmpSizes, OutSizes)
{
    if constexpr (util::TypeListSize<TmpSizes>::value == 0) { return util::TypeList<>{}; }
    else
    {
        using Index       = decltype(findMatchingOutput(util::TypeListElement_t<0, TmpSizes>{}, OutSizes{}));
        using TailIndices = decltype(mapTemporarySizes(tail(TmpSizes{}), dropByIndex(Index{}, OutSizes{})));
        return util::FuseTwo<util::TypeList<Index>, TailIndices>{};
    }
}

template<class T>
using SizeOf = std::integral_constant<std::size_t, sizeof(T)>;

template<class... T, std::size_t... Indices>
consteval auto addIndices(util::TypeList<T...>, std::index_sequence<Indices...>)
{
    return util::TypeList<util::TypeList<std::integral_constant<std::size_t, Indices>, T>...>{};
}

template<class T>
using AddIndices = decltype(addIndices(T{}, std::make_index_sequence<util::TypeListSize<T>::value>{}));

template<class Tmp, class Out>
using MapTemporarySizes = decltype(mapTemporarySizes(util::Map<SizeOf, Tmp>{}, AddIndices<util::Map<SizeOf, Out>>{}));

template<class... Indices, class... Tmp, class Output>
auto allocateOrMapTemporaries(const LocalIndex firstBody,
                              const LocalIndex lastBody,
                              util::TypeList<Indices...>,
                              std::tuple<Tmp...>,
                              const Output& output)
{
    auto allocOrMap = [&]<class Index, class T>(Index, T)
    {
        if constexpr (std::is_same_v<Index, None>)
        {
            auto holder = util::deviceAlloc<T[]>(lastBody - firstBody);
            return std::make_tuple(holder.get() - firstBody, std::move(holder));
        }
        else { return std::make_tuple(std::get<Index::value>(output), None{}); }
    };

    return std::make_tuple(allocOrMap(Indices{}, Tmp{})...);
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
template<class Config, class Tc, class Th, class Input, class... Out, class Interaction>
auto allocateTemporaries(LocalIndex firstBody,
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
                                      std::declval<Th*>(), std::declval<Input>(), std::declval<LocalIndex>()));
        using Result = decltype(unwrapModifiers(std::forward<Interaction>(interaction)(
            std::declval<ParticleData>(), std::declval<ParticleData>(), std::declval<Vec3<Tc>>(), std::declval<Tc>())));

        using PtrMap = MapTemporarySizes<Result, util::TypeList<Out...>>;

        auto ptrsAndHolders = allocateOrMapTemporaries(firstBody, lastBody, PtrMap{}, Result{}, output);

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
} // namespace cstone::ijloop::gpu_supercluster_nb_list_neighborhood_detail
