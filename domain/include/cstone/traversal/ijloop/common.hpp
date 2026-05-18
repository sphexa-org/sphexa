/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Common functionality for all ijloop neighborhood
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <tuple>
#include <limits>
#include <type_traits>

#include "cstone/sfc/box.hpp"
#include "cstone/traversal/boxoverlap.hpp"
#include "cstone/traversal/ijloop/ijloop.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/util/tuple_util.hpp"

namespace cstone::ijloop
{

// === particle data handling ===

template<class... Ts>
inline constexpr std::tuple<const Ts*...> makeConst(std::tuple<Ts*...> input)
{
    return {input};
}

template<class Tc, class ThP, class... Ts, class Th = std::remove_cvref_t<std::remove_pointer_t<ThP>>>
inline constexpr std::tuple<LocalIndex, Vec3<Tc>, Th, Ts...> loadParticleData(
    const Tc* x, const Tc* y, const Tc* z, const ThP h, std::tuple<const Ts*...> const& input, LocalIndex index)
{
#ifdef __CUDA_ARCH__
    auto load = [index](auto const* ptr) { return __ldg(&ptr[index]); };
#else
    auto load = [index](auto const* ptr) { return ptr[index]; };
#endif
    const Vec3<Tc> pos = {load(x), load(y), load(z)};
    Th hi;
    if constexpr (std::is_pointer_v<ThP>)
        hi = load(h);
    else
        hi = h;
    return std::tuple_cat(std::make_tuple(index, pos, hi), util::tupleMap(load, input));
}

template<class T>
inline constexpr std::remove_cvref_t<std::remove_pointer_t<T>> loadAtIndexIfPtr(T ptrOrConstant, LocalIndex index)
{
    if constexpr (std::is_pointer_v<T>)
        return ptrOrConstant[index];
    else
        return ptrOrConstant;
}

template<class... Ts>
inline constexpr void
storeParticleData(std::tuple<Ts*...> const& output, LocalIndex index, std::tuple<Ts...> const& value)
{
    util::for_each_tuple([index](auto* ptr, auto const& v) { ptr[index] = v; }, output, value);
}

template<class Tc, class ThP, class... Ts, class Th = std::remove_cvref_t<std::remove_pointer_t<ThP>>>
inline constexpr std::tuple<LocalIndex, Vec3<Tc>, Th, Ts...>
dummyParticleData(const Tc*, const Tc*, const Tc*, const ThP, std::tuple<const Ts*...> const&, LocalIndex index)
{
    constexpr Vec3<Tc> pos = {std::numeric_limits<Tc>::quiet_NaN(), std::numeric_limits<Tc>::quiet_NaN(),
                              std::numeric_limits<Tc>::quiet_NaN()};
    return std::make_tuple(index, pos, Th(0), Ts{}...);
}

template<class Tc, class Th, class... Ts>
inline constexpr bool requiresPbcHandling(Box<Tc> const& box, std::tuple<LocalIndex, Vec3<Tc>, Th, Ts...> const& iData)
{
    if ((box.boundaryX() != BoundaryType::periodic) && (box.boundaryY() != BoundaryType::periodic) &&
        (box.boundaryZ() != BoundaryType::periodic))
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

// === j reductions ===

namespace detail
{

template<class T>
inline constexpr void updateResultImpl(T& result, T const& value)
{
    result += value;
}

template<class T>
inline constexpr void updateResultImpl(reduction::min<T>& result, reduction::min<T> const& value)
{
    result.value = std::min(result.value, value.value);
}

template<class T>
inline constexpr void updateResultImpl(reduction::max<T>& result, reduction::max<T> const& value)
{
    result.value = std::max(result.value, value.value);
}

template<class T>
inline constexpr void updateResultImpl(symmetric::even<T>& result, symmetric::even<T> const& value)
{
    updateResultImpl(result.value, value.value);
}

template<class T>
inline constexpr void updateResultImpl(symmetric::odd<T>& result, symmetric::odd<T> const& value)
{
    updateResultImpl(result.value, value.value);
}

} // namespace detail

template<class... Ts>
inline constexpr void updateResult(std::tuple<Ts...>& result, std::tuple<Ts...> const& value)
{
    util::for_each_tuple([](auto& r, auto const& v) { detail::updateResultImpl(r, v); }, result, value);
}

namespace detail
{

template<class T>
struct IsSymmetric : std::false_type
{
};

template<class T>
struct IsSymmetric<symmetric::even<T>> : std::true_type
{
};

template<class T>
struct IsSymmetric<symmetric::odd<T>> : std::true_type
{
};

template<class T>
inline constexpr T const& applySymmetryImpl(T const& value)
{
    return value;
}

template<class T>
inline constexpr symmetric::odd<T> applySymmetryImpl(symmetric::odd<T> const& value)
{
    return {-value.value};
}

} // namespace detail

template<class T>
struct IsFullySymmetric;

template<class... Ts>
struct IsFullySymmetric<std::tuple<Ts...>> : std::conjunction<detail::IsSymmetric<std::remove_cvref_t<Ts>>...>
{
};

template<class... Ts>
inline constexpr auto selectSymmetric(std::tuple<Ts...> const& symmetricValue, std::tuple<Ts...> const& asymmetricValue)
{
    return util::tupleMap(
        []<class T>(T const& s, T const& a)
        {
            if constexpr (detail::IsSymmetric<T>::value)
                return detail::applySymmetryImpl(s);
            else
                return a;
        },
        symmetricValue, asymmetricValue);
}

namespace detail
{

template<class T>
inline constexpr T unwrapModifiersImpl(T const& result)
{
    return result;
}

template<class T>
inline constexpr auto unwrapModifiersImpl(reduction::min<T> const& result)
{
    return unwrapModifiersImpl(result.value);
}

template<class T>
inline constexpr auto unwrapModifiersImpl(reduction::max<T> const& result)
{
    return unwrapModifiersImpl(result.value);
}

template<class T>
inline constexpr auto unwrapModifiersImpl(symmetric::even<T> const& result)
{
    return unwrapModifiersImpl(result.value);
}

template<class T>
inline constexpr auto unwrapModifiersImpl(symmetric::odd<T> const& result)
{
    return unwrapModifiersImpl(result.value);
}

} // namespace detail

template<class... Ts>
inline constexpr auto unwrapModifiers(std::tuple<Ts...> const& value)
{
    return util::tupleMap([](auto const& v) { return detail::unwrapModifiersImpl(v); }, value);
}

} // namespace cstone::ijloop
