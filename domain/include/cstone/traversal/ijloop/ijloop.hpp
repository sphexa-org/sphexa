/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Neighborhood-independent public interface for ijloop
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <concepts>
#include <tuple>
#include <limits>

#include "cstone/execution.hpp"
#include "cstone/sfc/box.hpp"
#include "cstone/traversal/groups.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/tree/octree.hpp"

namespace cstone::ijloop
{

namespace symmetric
{

/*! Wrapper struct to mark a value as being evenly symmetric.
 *
 * This can be used to indicate that the contained value returned from a particle-particle interaction exhibits even
 * symmetry, i.e., f(i, j) == f(j, i).
 */
template<class T>
struct even
{
    T value = {};
};

/*! Wrapper struct to mark a value as being oddly symmetric.
 *
 * This can be used to indicate that the contained value returned from a particle-particle interaction exhibits odd
 * symmetry, i.e., f(i, j) == -f(j, i).
 */
template<class T>
struct odd
{
    T value = {};
};

} // namespace symmetric

namespace reduction
{

/*! Wrapper struct to mark a value as requiring a minimum reduction instead of the default sum.
 *
 * This can be used to indicate that in the neighbor reduction loop, the minimum of all neighbor values should be used
 * instead of the sum, i.e., f(i) = min(f(i, j) for all neighbors j).
 */
template<class T>
struct min
{
    T value = std::numeric_limits<T>::max();
};

/*! Wrapper struct to mark a value as requiring a maximum reduction instead of the default sum.
 *
 * This can be used to indicate that in the neighbor reduction loop, the maximum of all neighbor values should be used
 * instead of the sum, i.e., f(i) = max(f(i, j) for all neighbors j).
 */
template<class T>
struct max
{
    T value = std::numeric_limits<T>::lowest();
};

} // namespace reduction

namespace detail
{

template<class T>
constexpr auto unwrapModifiersImpl(T const& result)
{
    if constexpr (util::HasValueMember<T>)
        return unwrapModifiersImpl(result.value);
    else
        return result;
}

} // namespace detail

template<class... Ts>
constexpr auto unwrapModifiers(std::tuple<Ts...> const& value)
{
    return util::tupleMap([](auto const& v) { return detail::unwrapModifiersImpl(v); }, value);
}

namespace detail
{

template<class T>
inline constexpr bool IsTupleOfPointers_v = false;

template<class... Ts>
inline constexpr bool IsTupleOfPointers_v<std::tuple<Ts...>> =
    (std::is_pointer_v<Ts> && ...) && (std::is_trivially_copyable_v<std::remove_pointer_t<Ts>> && ...);
} // namespace detail

//! @brief Restricts types to std::tuples of pointers to trivially copyable types.
template<class T>
concept TupleOfPointers = detail::IsTupleOfPointers_v<T>;

template<class... Ts>
constexpr std::tuple<const Ts*...> makeConst(std::tuple<Ts*...> input)
{
    return {input};
}

namespace detail
{
/*! @brief Maps a std::tuple of pointers to a std::tuple of the pointee types. */
template<class T>
struct DereferencedTuple;

template<class... Ts>
requires TupleOfPointers<std::tuple<Ts...>> struct DereferencedTuple<std::tuple<Ts...>>
{
    using type = std::tuple<std::decay_t<std::remove_pointer_t<Ts>>...>;
};

struct EmptyPostamble
{
    template<class ParticleData, class Result>
    constexpr Result operator()(const ParticleData&, const Result& result) const
    {
        return result;
    }
};

} // namespace detail

//! @brief Concept satisfied by a floating point number, or a pointer to one. Used e.g. for smoothing lengths.
template<class T>
concept FpOrPtrToFp =
    (std::is_pointer_v<T> && std::is_floating_point_v<std::remove_pointer_t<T>>) || std::is_floating_point_v<T>;

struct Statistics
{
    const std::size_t numBodies, numBytes;
};

//! Empty postamble that does nothing. Should always be preferred over a custom empty postamble, as it enables certain
//! optimizations in the neighborhood implementations.
constexpr detail::EmptyPostamble empty_postamble;

//! @brief A tuple of stack variables with data associated with one particle
template<std::floating_point Tc, FpOrPtrToFp ThP, TupleOfPointers Input>
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
        postamble(i, unwrapModifiers(interaction(i, j, posdiff, r2)))        // must be callable with this signature
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
         FpOrPtrToFp ThP,
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
    using InteractionResultType = decltype(std::declval<Interaction>()(std::declval<ParticleDataType>(),
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

//! @brief Convenience factory to construct an @p IjLoopData with explicit Tc and ThP and deduced tuple/functor types.
template<class Tc, class ThP, class Input, class Output, class Interaction, class Postamble>
auto makeIjLoopData(const Input& in, const Output& out, const Interaction& interaction, const Postamble& postamble)
{
    auto constInput = makeConst(in);
    return IjLoopData<Tc, ThP, std::decay_t<decltype(constInput)>, std::decay_t<Output>, std::decay_t<Interaction>,
                      std::decay_t<Postamble>>{constInput, out, interaction, postamble};
}

namespace detail
{

struct ConceptTestInteraction
{
    constexpr std::tuple<int> operator()(std::tuple<LocalIndex, Vec3<double>, float>,
                                         std::tuple<LocalIndex, Vec3<double>, float>,
                                         Vec3<double>,
                                         float) const
    {
        return {0};
    }
};

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
    nb.build(exec, tree, box, totalBodies, groups, x, y, z, h);
    {
        nb.build(exec, tree, box, totalBodies, groups, x, y, z, h).stats()
    } -> std::same_as<Statistics>;
    {
        nb.build(exec, tree, box, totalBodies, groups, x, y, z, h)
            .ijLoop(IjLoopData<double, float*, std::tuple<>, std::tuple<int*>, detail::ConceptTestInteraction>{
                std::tuple(), std::tuple<int*>(), detail::ConceptTestInteraction{}, empty_postamble})
    } -> std::same_as<void>;
};

} // namespace detail

template<class T>
concept NeighborhoodBuilder =
    detail::NeighborhoodBuilder<T, execution::Cpu> || detail::NeighborhoodBuilder<T, execution::Gpu>;

} // namespace cstone::ijloop
