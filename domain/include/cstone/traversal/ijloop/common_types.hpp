/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Common types used in ijloop interfaces
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <cstddef>

namespace cstone::ijloop
{

struct Statistics
{
    const std::size_t numBodies, numBytes;
};

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

} // namespace cstone::ijloop
