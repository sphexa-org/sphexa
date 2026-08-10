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

/*! @brief Concept satisfied by a particle smoothing length.
 *
 * Accepts either a floating-point value (a uniform smoothing length shared by all particles) or a pointer to a
 * floating-point value (per-particle smoothing lengths). Both scalar and pointer forms let the neighborhood
 * implementation load the smoothing length of a particle at a given index.
 */
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

/*! @brief Concept for an input argument to the neighbor loop.
 *
 * Satisfied by a tuple of pointers to trivially-copyable types, where each pointer addresses the per-particle array
 * for one input property (e.g., masses, charges). The loop loads the element at each neighbor's index from every
 * pointer in the tuple.
 */
template<class T>
concept Input = detail::PointerTuple<T>;

/*! @brief Concept for an output argument to the neighbor loop.
 *
 * Satisfied by a tuple of pointers to trivially-copyable types, where each pointer addresses the per-particle output
 * array for one output property. The loop writes the accumulated interaction result for each particle through these
 * pointers.
 */
template<class T>
concept Output = detail::PointerTuple<T> && std::tuple_size_v<T> >= 1;

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

template<class T, class Tc, class ThP, class Inp>
concept InteractionCorrectlyCallable =
    requires(ParticleData<Tc, ThP, Inp> iData, ParticleData<Tc, ThP, Inp> jData, Vec3<Tc> ijPosDiff, Tc distSq)
{
    std::declval<T>()(iData, jData, ijPosDiff, distSq);
};

template<class T, class Tc, class ThP, class Inp>
concept InteractionReturnsValueTuple = requires(
    detail::ParticleData<Tc, ThP, Inp> iData, detail::ParticleData<Tc, ThP, Inp> jData, Vec3<Tc> ijPosDiff, Tc distSq)
{
    {
        std::declval<T>()(iData, jData, ijPosDiff, distSq)
    } -> ValueTuple;
};
} // namespace detail

/*! @brief Concept for a pairwise particle interaction functor.
 *
 * A type @p T satisfies @c Interaction if it is callable with the per-particle data of two particles (referred to as i
 * and j), the distance vector from i to j, and the squared distance between them. The return value must be a std::tuple
 * of values. Each element of the returned tuple represents the contribution of the pair (i, j) to the output properties
 * accumulated for particle i; elements may be wrapped in @ref symmetric::even, @ref symmetric::odd,
 * @ref reduction::min, or @ref reduction::max to select the accumulation strategy.
 */
template<class T, class Tc, class ThP, class Inp>
concept Interaction =
    std::floating_point<Tc> && H<ThP> && Input<Inp> && detail::InteractionCorrectlyCallable<T, Tc, ThP, Inp> &&
    detail::InteractionReturnsValueTuple<T, Tc, ThP, Inp>;

namespace detail
{
template<std::floating_point Tc, H ThP, Input Inp, Interaction<Tc, ThP, Inp> Inter>
using InteractionResult = decltype(std::declval<Inter>()(std::declval<ParticleData<Tc, ThP, Inp>>(),
                                                         std::declval<ParticleData<Tc, ThP, Inp>>(),
                                                         std::declval<Vec3<Tc>>(),
                                                         std::declval<Tc>()));

template<class T>
struct RemoveModifierFromElemT
{
    using type = T;
};

template<class T>
struct RemoveModifierFromElemT<symmetric::even<T>>
{
    using type = RemoveModifierFromElemT<T>::type;
};

template<class T>
struct RemoveModifierFromElemT<symmetric::odd<T>>
{
    using type = RemoveModifierFromElemT<T>::type;
};

template<class T>
struct RemoveModifierFromElemT<reduction::min<T>>
{
    using type = RemoveModifierFromElemT<T>::type;
};

template<class T>
struct RemoveModifierFromElemT<reduction::max<T>>
{
    using type = RemoveModifierFromElemT<T>::type;
};

template<ValueTuple T>
struct RemoveModifierT;

template<class... Ts>
struct RemoveModifierT<std::tuple<Ts...>>
{
    using type = std::tuple<typename RemoveModifierFromElemT<Ts>::type...>;
};

template<ValueTuple T>
using RemoveModifier = RemoveModifierT<T>::type;

template<PointerTuple T>
struct RemovePointersT;

template<class... Ts>
struct RemovePointersT<std::tuple<Ts*...>>
{
    using type = std::tuple<Ts...>;
};

template<class T, class Out>
concept ValidOutput =
    ValueTuple<T> && Output<Out> && std::same_as<RemoveModifier<T>, typename RemovePointersT<Out>::type>;

template<class T, class Tc, class ThP, class Inp, class Inter>
concept PostambleCorrectlyCallable =
    requires(ParticleData<Tc, ThP, Inp> iData, RemoveModifier<InteractionResult<Tc, ThP, Inp, Inter>> iResult)
{
    std::declval<T>()(iData, iResult);
};

template<class T, class Tc, class ThP, class Inp, class Out, class Inter>
concept PostambleReturnValueCompatibleWithOutput = requires(
    detail::ParticleData<Tc, ThP, Inp> iData, detail::RemoveModifier<InteractionResult<Tc, ThP, Inp, Inter>> iResult)
{
    {
        std::declval<T>()(iData, iResult)
    } -> ValidOutput<Out>;
};

template<class T, class Tc, class ThP, class Inp, class Out>
concept InteractionReturnValueCompatibleWithOutput = requires(
    detail::ParticleData<Tc, ThP, Inp> iData, detail::ParticleData<Tc, ThP, Inp> jData, Vec3<Tc> ijPosDiff, Tc distSq)
{
    {
        std::declval<T>()(iData, jData, ijPosDiff, distSq)
    } -> ValidOutput<Out>;
};

} // namespace detail

/*! @brief Concept for a postamble functor applied after the neighbor loop.
 *
 * A type @p T satisfies @c Postamble if it is callable with the per-particle data of a particle and the accumulated
 * interaction result (with modifier types stripped) for that particle. The return value must be a std::tuple whose
 * element types match the @ref Output argument after pointer removal. The postamble runs once per particle after all
 * neighbors have been visited and lets the caller transform the accumulated result before it is written to the output
 * arrays; use @ref cstone::ijloop::empty_postamble "empty_postamble" when no transformation is needed.
 */
template<class T, class Tc, class ThP, class Inp, class Out, class Inter>
concept Postamble = std::floating_point<Tc> && H<ThP> && Input<Inp> && Output<Out> &&
                    Interaction<Inter, Tc, ThP, Inp> && detail::PostambleCorrectlyCallable<T, Tc, ThP, Inp, Inter> &&
                    ((std::same_as<T, ::cstone::ijloop::detail::EmptyPostamble> &&
                      detail::InteractionReturnValueCompatibleWithOutput<Inter, Tc, ThP, Inp, Out>) ||
                     detail::PostambleReturnValueCompatibleWithOutput<T, Tc, ThP, Inp, Out, Inter>);

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

/*! @brief Concept for a neighborhood search backend.
 *
 * A type @p T satisfies @c Neighborhood if it exposes a @c stats() method returning @ref Statistics and an
 * @c ijLoop(input, output, interaction, postamble) method matching the @ref Input, @ref Output, @ref Interaction, and
 * @ref Postamble concepts. The @c ijLoop iterates over all particle pairs (i, j) within each other's smoothing
 * length, invokes the interaction functor for each pair, accumulates the results, and finally calls the postamble
 * once per particle before writing to the output arrays.
 */
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

/*! @brief Concept for a factory that builds a neighborhood search backend.
 *
 * A type @p T satisfies @c NeighborhoodBuilder if its @c build(exec, tree, box, totalBodies, groups, x, y, z, h)
 * method returns an object satisfying @ref Neighborhood. The build method is instantiated for both CPU and GPU
 * execution policies; @c x, @c y, @c z are the particle coordinate arrays, @c h is the smoothing-length array,
 * @c tree is the octree view, @c box the simulation domain, @c totalBodies the particle count, and @c groups the
 * target particle grouping.
 */
template<class T>
concept NeighborhoodBuilder =
    detail::NeighborhoodBuilder<T, execution::Cpu> || detail::NeighborhoodBuilder<T, execution::Gpu>;
} // namespace cstone::ijloop::concepts
