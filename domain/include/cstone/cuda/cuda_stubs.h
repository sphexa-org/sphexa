/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  CUDA/Thrust stubs to provide declarations without definitions for use in non-CUDA builds
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <type_traits>
#include <vector>

template<class T, class Alloc>
T* rawPtr(std::vector<T, Alloc>& p)
{
    return p.data();
}

template<class T, class Alloc>
const T* rawPtr(const std::vector<T, Alloc>& p)
{
    return p.data();
}

template<class T>
void memcpyH2D(const T* src, std::size_t n, T* dest);

template<class T>
void memcpyD2H(const T* src, std::size_t n, T* dest);

template<class T>
void memcpyD2D(const T* src, std::size_t n, T* dest);

void syncGpu();

/*! @brief detection trait to determine whether a template parameter is a device vector
 *
 * @tparam Vector the Vector type to check
 *
 * Add specializations for each type of vector that should be recognized as on device
 */
template<class Vector>
struct IsDeviceVector : public std::false_type
{
};
