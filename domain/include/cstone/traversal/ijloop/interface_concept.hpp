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

} // namespace detail

/*! @brief Restricts types to std::tuples of pointers to trivially copyable types. */
template<class T>
concept TupleOfPointers = detail::IsTupleOfPointers<T>::value;


/*
template<class Tc, class ThP, class Input, class Output, class Interaction, class Postamble, class Reduction>
struct Types
{
    using ParticleData = decltype(loadParticleData(std::declval<const Tc*>(),
                                                   std::declval<const Tc*>(),
                                                   std::declval<const Tc*>(),
                                                   std::declval<ThP>(),
                                                   makeConst(std::declval<Input>()),
                                                   std::declval<LocalIndex>()));
    using Result       = decltype(std::declval<Interaction>()(
        std::declval<ParticleData>(), std::declval<ParticleData>(), std::declval<Vec3<Tc>>(), std::declval<Tc>()));
    using PostambleResult =
        decltype(std::declval<Postamble>()(std::declval<ParticleData>(), unwrapModifiers(std::declval<Result>())));
    using ReductionResult          = decltype(std::declval<Reduction>()(std::declval<ParticleData>(),
                                                               unwrapModifiers(std::declval<Result>()),
                                                               unwrapModifiers(std::declval<PostambleResult>())));
    using UnwrappedReductionResult = decltype(unwrapModifiers(std::declval<ReductionResult>()));
};

} // namespace detail

template<class Tc,
         class ThP,
         class Input,
         class Output,
         class Interaction,
         class Postamble = detail::EmptyPostamble,
         class Reduction = detail::NoReduction>
consteval detail::Types<Tc,
                        ThP,
                        std::decay_t<Input>,
                        std::decay_t<Output>,
                        std::decay_t<Interaction>,
                        std::decay_t<Postamble>,
                        std::decay_t<Reduction>>
types(const Tc*,
      const Tc*,
      const Tc*,
      ThP,
      Input,
      Output,
      Interaction,
      Postamble = empty_postamble,
      Reduction = no_reduction)
{
    return {};
}
 */

/*! A dataset that can be passed to an ijLoop.
 *
 * The associated types are:
 * @tparam Input
 * @tparam Output
 * @tparam Interaction
 * @tparam Postamble
 * @tparam Reduction
 *
 * These types are subject to restrictions, modelled by a concept
 */
template<std::floating_point Tc,
         TupleOfPointers Input,
         TupleOfPointers Output,
         class Interaction,
         class Postamble,
         class Reduction>
struct IJLoopDataset
{
    /*! typedefs:
     *
     * Fixed with neighborhood type
     *
     * Tc     types of x,y,z
     * ThP    type of h or pointer to h
     *
     * ------------
     *
     * ParticleData:     the tuple input data for a single particle in an i-j interaction,
     *                   tuple_cat(std::tuple<LocalIndex, Vec3<Tc>, Th>, *Input),
     *                   part of the interface specification of the Interaction signature
     *
     * Result:          return type of Interaction(ParticleData, ParticleData, Vec3<Tc>, Tc)
     *                  what a single i-j interaction returns
     *
     * PostambleResult:  return type of Postamble(ParticleData, Result)
     *
     * ReductionResult:  return type of Reduction(ParticleData, Result, PostambleResult)
     *
     */
    using ParticleData = std::tuple<LocalIndex, Vec3<Tc>>;

    Input input;             // tuple of pointers to trivial types
    Output output;           // tuple of pointers to trivial types

    Interaction interaction; // callable, signature must accept (ParticleData, ParticleData, Vec3<Tc>, Th r2)

    Postamble postamble;     // callable, signature must accept (ParticleData, Result)
                             // return type must be (tuple with dereferenced types of Output)

    Reduction reduction;     // callable, signature must accept (ParticleData, Result, PostambleResult)
                             // return type must be trivially copiable
};

} // namespace cstone::ijloop
