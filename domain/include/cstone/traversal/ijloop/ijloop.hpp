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

struct EmptyPostamble
{
    template<class ParticleData, class Result>
    constexpr Result operator()(ParticleData const&, Result const& result) const
    {
        return result;
    }
};

} // namespace detail

//! Empty postamble that does nothing. Should always be prefered over a custom empty postamble, as it enables certain
//! optimizations in the neighborhood implementations.
constexpr detail::EmptyPostamble empty_postamble;

struct Statistics
{
    const std::size_t numBodies, numBytes;
};

namespace detail
{

struct ConceptTestInteraction
{
    constexpr std::tuple<int>
    operator()(std::tuple<LocalIndex, double, float>, std::tuple<LocalIndex, double, float>, Vec3<double>, double) const
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
            .ijLoop(std::tuple(), std::tuple<int*>(), detail::ConceptTestInteraction{}, empty_postamble)
    } -> std::same_as<void>;
};

} // namespace detail

template<class T>
concept NeighborhoodBuilder =
    detail::NeighborhoodBuilder<T, execution::Cpu> || detail::NeighborhoodBuilder<T, execution::Gpu>;

} // namespace cstone::ijloop
