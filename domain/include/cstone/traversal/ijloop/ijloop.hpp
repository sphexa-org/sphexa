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
 * @brief Neighborhood-independent public interface for ijloop
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <concepts>
#include <tuple>
#include <limits>

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

} // namespace detail

template<class T>
concept Neighborhood = requires(T nb,
                                OctreeNsView<double, unsigned> tree,
                                Box<double> box,
                                LocalIndex totalBodies,
                                GroupView groups,
                                const double* x,
                                const double* y,
                                const double* z,
                                const float* h)
{
    nb.build(tree, box, totalBodies, groups, x, y, z, h);
    {
        nb.build(tree, box, totalBodies, groups, x, y, z, h).stats()
    } -> std::same_as<Statistics>;
    {
        nb.build(tree, box, totalBodies, groups, x, y, z, h)
            .ijLoop(std::tuple(), std::tuple<int*>(), detail::ConceptTestInteraction{}, empty_postamble)
    } -> std::same_as<void>;
};

} // namespace cstone::ijloop
