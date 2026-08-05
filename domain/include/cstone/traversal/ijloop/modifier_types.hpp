/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Wrapper types to define symmetry and reduction kinds
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <limits>

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

} // namespace cstone::ijloop
