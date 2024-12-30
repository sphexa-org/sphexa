/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Reallocation of thrust device vectors in a separate compilation unit for use from .cpp code
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/util/tuple_util.hpp"

//! @brief resizes a vector with a determined growth rate upon reallocation
template<class Vector>
void reallocate(Vector& vector, size_t size, double growthRate)
{
    size_t current_capacity = vector.capacity();

    if (size > current_capacity)
    {
        size_t reserve_size = double(size) * growthRate;
        vector.reserve(reserve_size);
    }
    vector.resize(size);
}

//! @brief if reallocation of the underlying buffer is necessary, first deallocate it
template<class Vector>
void reallocateDestructive(Vector& vector, size_t size, double growthRate)
{
    if (size > vector.capacity())
    {
        // swap with an empty temporary to force deallocation
        Vector().swap(vector);
    }
    reallocate(vector, size, growthRate);
}

template<class... Arrays>
void reallocate(std::size_t size, double growthRate, Arrays&... arrays)
{
    [[maybe_unused]] std::initializer_list<int> list{(reallocate(arrays, size, growthRate), 0)...};
}

/*! @brief resize a vector to given number of bytes if current size is smaller
 *
 * @param[inout] vec       an STL or thrust-like vector
 * @param[in]    numBytes  minimum buffer size in bytes of @a vec
 * @return                 number of elements (vec.size(), not bytes) of supplied argument vector
 *
 * Note: does not decrease the size of @p vec
 */
template<class Vector>
size_t reallocateBytes(Vector& vec, size_t numBytes, double growthRate)
{
    constexpr size_t elementSize = sizeof(typename Vector::value_type);
    size_t originalSize          = vec.size();

    size_t currentSizeBytes = originalSize * elementSize;
    if (currentSizeBytes < numBytes) { reallocate(vec, (numBytes + elementSize - 1) / elementSize, growthRate); }

    return originalSize;
}

//! @brief reallocate memory by first deallocating all scratch to reduce fragmentation and decrease temp mem footprint
template<class... Vectors1, class... Vectors2>
void lowMemReallocate(size_t size,
                      float growthRate,
                      std::tuple<Vectors1&...> conserved,
                      std::tuple<Vectors2&...> scratch)
{
    // if the new size exceeds capacity, we first deallocate all scratch buffers to make space for the reallocations
    util::for_each_tuple(
        [size](auto& v)
        {
            if (size > v.capacity()) { std::decay_t<decltype(v)>{}.swap(v); }
        },
        scratch);
    util::for_each_tuple([size, growthRate](auto& v) { reallocate(v, size, growthRate); }, conserved);
    util::for_each_tuple([size, growthRate](auto& v) { reallocate(v, size, growthRate); }, scratch);
}
