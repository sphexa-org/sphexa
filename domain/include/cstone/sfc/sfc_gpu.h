/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  SFC encoding/decoding in 32- and 64-bit on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/execution.hpp"
#include "cstone/sfc/sfc.hpp"

namespace cstone
{

/*! @brief CUDA kernel to compute SFC (Hilbert/Morton) keys from integer coordinates
 *
 * @tparam     KeyType    32-bit or 64-bit SFC key type
 * @tparam     T          floating-point type for coordinates
 * @param[in]  exec       CUDA stream to launch kernel on
 * @param[in]  x          input integer x-coordinates
 * @param[in]  y          input integer y-coordinates
 * @param[in]  z          input integer z-coordinates
 * @param[out] keys       output array of SFC keys
 * @param[in]  numKeys    number of keys to compute
 * @param[in]  box        bounding box for the coordinates
 */
template<class KeyType, class T>
extern void computeSfcKeys(
    execution::Gpu exec, const T* x, const T* y, const T* z, KeyType* keys, size_t numKeys, const Box<T>& box);

} // namespace cstone
