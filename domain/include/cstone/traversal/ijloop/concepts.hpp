/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Concepts for the neighborhood API
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <concepts>
#include <tuple>

#include "cstone/execution.hpp"
#include "cstone/sfc/box.hpp"
#include "cstone/traversal/groups.hpp"
#include "cstone/traversal/ijloop/common_types.hpp"
#include "cstone/traversal/ijloop/modifier_types.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/tree/octree.hpp"

namespace cstone::ijloop::concepts
{

template<class T>
concept H = (std::is_pointer_v<T> && std::is_floating_point_v<std::remove_pointer_t<T>>) || std::is_floating_point_v<T>;

namespace detail
{
template<class T>
constexpr bool isPointerTuple = false;

template<class... T>
constexpr bool isPointerTuple<std::tuple<T*...>> = (... && std::is_trivially_copyable_v<T>);

template<class T>
concept PointerTuple = isPointerTuple<T>;

template<class T>
constexpr bool isValueTuple = false;

template<class... T>
constexpr bool isValueTuple<std::tuple<T...>> = (... && (!std::is_pointer_v<T> && std::is_trivially_copyable_v<T>));

template<class T>
concept ValueTuple = isValueTuple<T>;

} // namespace detail

template<class T>
concept Input = detail::PointerTuple<T>;

template<class T>
concept Output = detail::PointerTuple<T>;

namespace detail
{

template<std::floating_point Tc, H ThP, Input Inp>
struct ParticleDataT;

template<std::floating_point Tc, H ThP, class... Ts>
struct ParticleDataT<Tc, ThP, std::tuple<Ts*...>>
{
    using type =
        std::tuple<LocalIndex, Vec3<Tc>, std::remove_cvref_t<std::remove_pointer_t<ThP>>, std::remove_const_t<Ts>...>;
};

template<std::floating_point Tc, H ThP, Input Inp>
using ParticleData = ParticleDataT<Tc, ThP, Inp>::type;
} // namespace detail

template<class T, class Tc, class ThP, class Inp>
concept Interaction = std::floating_point<Tc> && H<ThP> && Input<Inp> &&
                      requires(detail::ParticleData<Tc, ThP, Inp> iData,
                               detail::ParticleData<Tc, ThP, Inp> jData,
                               Vec3<Tc> ijPosDiff,
                               Tc distSq)
{
    {
        std::declval<T>()(iData, jData, ijPosDiff, distSq)
    } -> detail::ValueTuple;
};

namespace detail
{
template<std::floating_point Tc, H ThP, Input Inp, Interaction<Tc, ThP, Inp> Inter>
using InteractionResult = decltype(std::declval<Inter>()(std::declval<ParticleData<Tc, ThP, Inp>>(),
                                                         std::declval<ParticleData<Tc, ThP, Inp>>(),
                                                         std::declval<Vec3<Tc>>(),
                                                         std::declval<Tc>()));

template<class T>
struct RemoveWrapperFromElemT
{
    using type = T;
};

template<class T>
struct RemoveWrapperFromElemT<symmetric::even<T>>
{
    using type = RemoveWrapperFromElemT<T>::type;
};

template<class T>
struct RemoveWrapperFromElemT<symmetric::odd<T>>
{
    using type = RemoveWrapperFromElemT<T>::type;
};

template<class T>
struct RemoveWrapperFromElemT<reduction::min<T>>
{
    using type = RemoveWrapperFromElemT<T>::type;
};

template<class T>
struct RemoveWrapperFromElemT<reduction::max<T>>
{
    using type = RemoveWrapperFromElemT<T>::type;
};

template<ValueTuple T>
struct RemoveWrapperT;

template<class... Ts>
struct RemoveWrapperT<std::tuple<Ts...>>
{
    using type = std::tuple<typename RemoveWrapperFromElemT<Ts>::type...>;
};

template<ValueTuple T>
using RemoveWrapper = RemoveWrapperT<T>::type;

template<PointerTuple T>
struct RemovePointersT;

template<class... Ts>
struct RemovePointersT<std::tuple<Ts*...>>
{
    using type = std::tuple<Ts...>;
};

template<class T, class Out>
concept ValidOutput =
    ValueTuple<T> && Output<Out> && std::same_as<RemoveWrapper<T>, typename RemovePointersT<Out>::type>;

} // namespace detail

template<class T, class Tc, class ThP, class Inp, class Out, class Inter>
concept Postamble =
    std::floating_point<Tc> && H<ThP> && Input<Inp> && Output<Out> && Interaction<Inter, Tc, ThP, Inp> &&
    requires(detail::ParticleData<Tc, ThP, Inp> iData,
             detail::RemoveWrapper<detail::InteractionResult<Tc, ThP, Inp, Inter>> iResult)
{
    {
        std::declval<T>()(iData, iResult)
    } -> detail::ValidOutput<Out>;
};

namespace detail
{

struct ConceptTestInteraction
{
    constexpr std::tuple<int> operator()(std::tuple<LocalIndex, Vec3<double>, float, float>,
                                         std::tuple<LocalIndex, Vec3<double>, float, float>,
                                         Vec3<double>,
                                         double) const
    {
        return {0};
    }
};

struct ConceptTestPostamble
{
    constexpr std::tuple<double> operator()(std::tuple<LocalIndex, Vec3<double>, float, float>, std::tuple<int>) const
    {
        return {0.0};
    }
};

} // namespace detail

template<class T>
concept Neighborhood = requires(T nb,
                                const detail::ConceptTestInteraction interaction,
                                const detail::ConceptTestPostamble postamble,
                                std::tuple<float*> input,
                                std::tuple<double*> output)
{
    {
        nb.stats()
    } -> std::same_as<Statistics>;
    {
        nb.ijLoop(input, output, interaction, postamble)
    } -> std::same_as<void>;
};

namespace detail
{

template<class T, class Exec>
concept NeighborhoodBuilder = execution::Policy<Exec> && requires(Exec exec,
                                                                  T nb,
                                                                  OctreeNsView<double, unsigned> tree,
                                                                  Box<double> box,
                                                                  LocalIndex totalBodies,
                                                                  GroupView groups,
                                                                  const double* x,
                                                                  const double* y,
                                                                  const double* z,
                                                                  const float* h)
{
    {
        nb.build(exec, tree, box, totalBodies, groups, x, y, z, h)
    } -> Neighborhood;
};

} // namespace detail

template<class T>
concept NeighborhoodBuilder =
    detail::NeighborhoodBuilder<T, execution::Cpu> || detail::NeighborhoodBuilder<T, execution::Gpu>;
} // namespace cstone::ijloop::concepts
