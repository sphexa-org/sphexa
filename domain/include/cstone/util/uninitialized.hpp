/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Helper to get uninitialized memory for any type. Useful for uninitialized __shared__ buffers of arbitrary
 * types.
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <type_traits>

namespace util
{

template<class T>
class Uninitialized
{
    alignas(std::alignment_of_v<T>) unsigned char buffer[sizeof(T)];

public:
    constexpr std::remove_extent_t<T>* data() { return reinterpret_cast<std::remove_extent_t<T>*>(buffer); }

    constexpr std::remove_extent_t<T> const* data() const
    {
        return reinterpret_cast<std::remove_extent_t<T> const*>(buffer);
    }
};

} // namespace util
