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
 * @brief Basic functionality for looping over particles and neighbors
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <tuple>
#include <type_traits>

#include "cstone/sfc/box.hpp"
#include "cstone/traversal/groups.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/tree/octree.hpp"
#include "cstone/util/tuple_util.hpp"

namespace cstone::ijloop
{

namespace symmetry
{
struct Asymmetric
{
};
struct Even
{
};
struct Odd
{
};

inline constexpr Asymmetric asymmetric = {};
inline constexpr Even even             = {};
inline constexpr Odd odd               = {};
} // namespace symmetry

template<class... Ts>
inline constexpr std::tuple<const Ts*...> makeConstRestrict(std::tuple<Ts*...> input)
{
    // TODO: __restrict__
    return {input};
}

template<class Tc, class Th, class... Ts>
inline constexpr std::tuple<LocalIndex, Vec3<Tc>, Th, Ts...> loadParticleData(
    const Tc* x, const Tc* y, const Tc* z, const Th* h, std::tuple<const Ts*...> const& input, LocalIndex index)
{
    const Vec3<Tc> pos = {x[index], y[index], z[index]};
    return std::tuple_cat(std::make_tuple(index, pos, h[index]),
                          util::tupleMap([index](auto const* ptr) { return ptr[index]; }, input));
}

template<class Tc, class Th, class... Ts>
inline constexpr std::tuple<LocalIndex, Vec3<Tc>, Th, Ts...>
dummyParticleData(const Tc*, const Tc*, const Tc*, const Th*, std::tuple<const Ts*...> const&, LocalIndex index)
{
    constexpr Vec3<Tc> pos = {std::numeric_limits<Tc>::quiet_NaN(), std::numeric_limits<Tc>::quiet_NaN(),
                              std::numeric_limits<Tc>::quiet_NaN()};
    return std::make_tuple(index, pos, Th(0), Ts{}...);
}

template<class Tc, class Th, class... Ts>
inline constexpr bool requiresPbcHandling(Box<Tc> const& box, std::tuple<LocalIndex, Vec3<Tc>, Th, Ts...> const& iData)
{
    if (box.boundaryX() != BoundaryType::periodic & box.boundaryY() != BoundaryType::periodic &
        box.boundaryZ() != BoundaryType::periodic)
        return false;
    const Vec3<Tc>& iPos = std::get<1>(iData);
    const Tc twoHi       = Tc(2) * std::get<2>(iData);
    return !insideBox(iPos, {twoHi, twoHi, twoHi}, box);
}

template<class Tc, class Th, class... Ts>
inline constexpr std::tuple<Vec3<Tc>, Tc> posDiffAndDistSq(bool usePbc,
                                                           Box<Tc> const& box,
                                                           std::tuple<LocalIndex, Vec3<Tc>, Th, Ts...> const& iData,
                                                           std::tuple<LocalIndex, Vec3<Tc>, Th, Ts...> const& jData)
{
    const Vec3<Tc>& iPos = std::get<1>(iData);
    const Vec3<Tc>& jPos = std::get<1>(jData);
    Vec3<Tc> ijPosDiff   = iPos - jPos;
    if (usePbc)
    {
        ijPosDiff[0] -= (box.boundaryX() == BoundaryType::periodic) * box.lx() * std::rint(ijPosDiff[0] * box.ilx());
        ijPosDiff[1] -= (box.boundaryY() == BoundaryType::periodic) * box.ly() * std::rint(ijPosDiff[1] * box.ily());
        ijPosDiff[2] -= (box.boundaryZ() == BoundaryType::periodic) * box.lz() * std::rint(ijPosDiff[2] * box.ilz());
    }
    return {ijPosDiff, norm2(ijPosDiff)};
}

template<class Tc, class Th, class... Ts>
inline constexpr Th radiusSq(std::tuple<LocalIndex, Vec3<Tc>, Th, Ts...> const& data)
{
    return Th(4) * std::get<2>(data) * std::get<2>(data);
}

template<class... Ts>
inline constexpr void updateResult(std::tuple<Ts...>& result, std::tuple<Ts...> const& value)
{
    util::for_each_tuple([](auto& r, auto const& v) { r += v; }, result, value);
}

template<class... Ts>
inline constexpr void
storeParticleData(std::tuple<Ts*...> const& output, LocalIndex index, std::tuple<Ts...> const& value)
{
    util::for_each_tuple([index](auto* ptr, auto const& v) { ptr[index] = v; }, output, value);
}

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

struct Statistics
{
    const std::size_t numBodies, numBytes;
};

template<class T>
concept Symmetry =
    std::is_same_v<T, symmetry::Asymmetric> || std::is_same_v<T, symmetry::Even> || std::is_same_v<T, symmetry::Odd>;

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
            .ijLoop(std::tuple(), std::tuple<int*>(), detail::ConceptTestInteraction{}, symmetry::asymmetric)
    } -> std::same_as<void>;
};

} // namespace cstone::ijloop
