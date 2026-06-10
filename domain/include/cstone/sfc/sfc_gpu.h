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

template<class KeyType, class T>
extern void computeSfcKeysGpu(
    const T* x, const T* y, const T* z, KeyType* keys, size_t numKeys, const Box<T>& box, execution::Gpu exec);

template<class KeyType, class T>
inline void computeSfcKeys(
    execution::Gpu exec, const T* x, const T* y, const T* z, KeyType* keys, size_t numKeys, const Box<T>& box)
{
    computeSfcKeysGpu(x, y, z, keys, numKeys, box, exec);
}

template<class KeyType, class T>
inline void
computeSfcKeys(execution::Cpu, const T* x, const T* y, const T* z, KeyType* keys, size_t numKeys, const Box<T>& box)
{
    computeSfcKeys(x, y, z, keys, numKeys, box);
}

} // namespace cstone
