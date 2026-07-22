/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! \file
 * \brief  Helpers for handling smoothing length data
 *
 * \author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <type_traits>

#include "cstone/tree/definitions.h"

namespace util
{

template<class T>
inline constexpr std::remove_cvref_t<std::remove_pointer_t<T>> loadAtIndexIfPtr(T ptrOrValue, cstone::LocalIndex index)
{
    if constexpr (std::is_pointer_v<T>)
        return ptrOrValue[index];
    else
        return ptrOrValue;
}

template<class T>
inline constexpr T infToZero(T value)
{ return std::isinf(value) ? 0 : value; }

} // namespace util
