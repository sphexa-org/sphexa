/*
 * Cornerstone octree
 *
 * Copyright (c) 2026 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Neighbor search on GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <tuple>
#include <type_traits>

#include "cstone/tree/definitions.h"
#include "cstone/traversal/ijloop/ijloop.hpp"

namespace cstone::ijloop
{

namespace detail
{

template<class T>
struct IsTupleOfPointers : std::false_type
{
};

template<class... Ts>
struct IsTupleOfPointers<std::tuple<Ts...>>
    : std::bool_constant<(std::is_pointer_v<Ts> && ...) &&
                         (std::is_trivially_copyable_v<std::remove_pointer_t<Ts>> && ...)>
{
};

/*! @brief Maps a std::tuple of pointers to a std::tuple of the pointee types. */
template<class T>
struct DereferencedTuple;

template<class... Ts>
struct DereferencedTuple<std::tuple<Ts...>>
{
    using type = std::tuple<std::decay_t<std::remove_pointer_t<Ts>>...>;
};

} // namespace detail

//! @brief Restricts types to std::tuples of pointers to trivially copyable types.
template<class T>
concept TupleOfPointers = detail::IsTupleOfPointers<T>::value;

//! @brief A tuple of stack variables with data associated with one particle
template<std::floating_point Tc, class ThP, TupleOfPointers Input>
using ParticleData =
    decltype(std::tuple_cat(std::declval<std::tuple<LocalIndex, Vec3<Tc>, std::remove_pointer_t<ThP>>>(),
                            std::declval<typename detail::DereferencedTuple<Input>::type>()));

//! @brief Defines what a valid i-j pair interaction is: a function object taking (i, j, posdiff, r2)
template<class F, class Tc, class ThP, class Input>
concept PairInteraction = requires(const F& func,
                                   const ParticleData<Tc, ThP, Input>& i,
                                   const ParticleData<Tc, ThP, Input>& j,
                                   Vec3<Tc> posdiff,
                                   std::remove_pointer_t<ThP> r2)
{
    {func(i, j, posdiff, r2)}; // must be callable with this signature
};

//! @brief A postamble is callable with (ParticleData, interaction result), and returns a tuple compatible with Output
template<class Postamble, class Interaction, class Tc, class ThP, class Input, class Output>
concept ValidPostamble = PairInteraction<Interaction, Tc, ThP, Input> && requires(const Postamble& postamble,
                                                                                  const Interaction& interaction,
                                                                                  const ParticleData<Tc, ThP, Input>& i,
                                                                                  const ParticleData<Tc, ThP, Input>& j,
                                                                                  Vec3<Tc> posdiff,
                                                                                  std::remove_pointer_t<ThP> r2)
{
    {
        postamble(i, interaction(i, j, posdiff, r2))                         // must be callable with this signature
        } -> std::same_as<typename detail::DereferencedTuple<Output>::type>; // must return this type
};

/*! A dataset that can be passed to an ijLoop.
 *
 * @tparam Tc              types of x,y,z coordinates
 * @tparam ThP             type of h, pointer to floating_point if search radius per particle is variable
 * @tparam Input           tuple of input particle field pointers
 * @tparam Output          tuple of output particle field pointers
 * @tparam Interaction     function object satisfying the PairInteraction concept
 * @tparam Postamble       function object satisfying the ValidPostamble concept
 */
template<std::floating_point Tc,
         class ThP,
         TupleOfPointers Input,
         TupleOfPointers Output,
         PairInteraction<Tc, ThP, Input> Interaction,
         ValidPostamble<Interaction, Tc, ThP, Input, Output> Postamble = detail::EmptyPostamble>
struct IjLoopData
{
    //! @brief The tuple input data for a single particle in an i-j interaction,
    using ParticleDataType = ParticleData<Tc, ThP, Input>;

    //! @brief Type of search radii, e.g. smoothing lengths
    using RadiusType = std::remove_pointer_t<ThP>;

    //! @brief What an i-j interaction returns
    using Result = decltype(std::declval<Interaction>()(std::declval<ParticleDataType>(),
                                                        std::declval<ParticleDataType>(),
                                                        std::declval<Vec3<Tc>>(),
                                                        std::declval<RadiusType>()));

    //! @brief what the postamble returns - will be stored back to the output fields
    using PostambleResultType = typename detail::DereferencedTuple<Output>::type;

    Input input;
    Output output;

    //! @brief the i-j interaction kernel
    Interaction interaction;
    //! @brief Post-processing to apply to the Result after the j-loop
    Postamble postamble;
};

} // namespace cstone::ijloop
