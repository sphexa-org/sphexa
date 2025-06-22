/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Basic math
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cstdint>

#include "cstone/cuda/annotation.hpp"
#include "clz.hpp"

namespace cstone
{

//! @brief ceil(dividend/divisor) for unsigned integers
HOST_DEVICE_FUN constexpr size_t iceil(size_t dividend, unsigned divisor)
{
    return (dividend + divisor - 1lu) / divisor;
}

//! @brief round up @p n to multiple of @p m
HOST_DEVICE_FUN constexpr size_t round_up(size_t n, unsigned m) { return ((n + m - 1) / m) * m; }

//! @brief compute 1 2 1 3 1 2 1 4 1 2 1, ... pattern, e.g. for "butterfly" shflXor reductions
HOST_DEVICE_FUN constexpr uint32_t butterfly(uint32_t i)
{
    if (i == 0) { return 0; }
    return 1 + countTrailingZeros(i);
}

} // namespace cstone
