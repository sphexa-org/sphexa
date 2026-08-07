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

#include <functional>
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

/*! @brief Restricts types to std::tuples of pointers to trivially copyable types. */
template<class T>
concept TupleOfPointers = detail::IsTupleOfPointers<T>::value;

template<std::floating_point Tc, class ThP, TupleOfPointers Input>
using ParticleData =
    decltype(std::tuple_cat(std::declval<std::tuple<LocalIndex, Vec3<Tc>, std::remove_pointer_t<ThP>>>(),
                            std::declval<typename detail::DereferencedTuple<Input>::type>()));

template<class F, class Tc, class ThP, class Input>
concept PairInteraction = requires(const F& func,
                                           const ParticleData<Tc, ThP, Input>& i,
                                           const ParticleData<Tc, ThP, Input>& j,
                                           Vec3<Tc> posdiff,
                                           std::remove_pointer_t<ThP> r2)
{
    {func(i, j, posdiff, r2)};
};

template<class Postamble, class Interaction, class Tc, class ThP, class Input, class Output>
concept ValidPostamble = PairInteraction<Interaction, Tc, ThP, Input> && requires(const Postamble& postamble,
                                                                                  const Interaction& interaction,
                                                                                  const ParticleData<Tc, ThP, Input>& i,
                                                                                  const ParticleData<Tc, ThP, Input>& j,
                                                                                  Vec3<Tc> posdiff,
                                                                                  std::remove_pointer_t<ThP> r2)
{
    {
        postamble(i, interaction(i, j, posdiff, r2))
        } -> std::same_as<typename detail::DereferencedTuple<Output>::type>;
};

/*! A dataset that can be passed to an ijLoop.
 *
 * @tparam Tc              types of x,y,z coordinates
 * @tparam ThP             type of h, pointer to floating_point if search radius per particle is variable
 * @tparam Input           tuple of input particle field pointers
 * @tparam Output          tuple of output particle field pointers
 * @tparam Interaction     the i-j interaction
 * @tparam Postamble       post-processing to apply to the i-j interaction result after reduction over j
 */
template<std::floating_point Tc,
         class ThP,
         TupleOfPointers Input,
         TupleOfPointers Output,
         PairInteraction<Tc, ThP, Input> Interaction,
         ValidPostamble<Interaction, Tc, ThP, Input, Output> Postamble = detail::EmptyPostamble>
struct IJLoopDataset
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

    //! @brief tuple of pointers to input particle fields
    Input input;
    //! @brief tuple of pointers to output particle fields
    Output output;

    /*! @brief the i-j interaction kernel
     *
     * callable, signature must accept (ParticleData, ParticleData, Vec3<Tc>, Th r2)
     * return type must match the Result parameter of the Postamble call signature
     */
    Interaction interaction;

    /*! @brief Post-processing to apply to the Result after the j-loop
     *
     * callable, signature must accept (ParticleData, Result), return type must be PostambleResultType
     */
    Postamble postamble;
};

} // namespace cstone::ijloop
