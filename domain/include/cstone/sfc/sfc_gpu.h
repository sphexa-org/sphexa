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

#include "cstone/sfc/sfc.hpp"

namespace cstone
{

template<class KeyType, class T>
extern void computeSfcKeysGpu(const T* x, const T* y, const T* z, KeyType* keys, size_t numKeys, const Box<T>& box);

template<bool useGpu, class KeyType, class T>
void computeSfcKeys(const T* x, const T* y, const T* z, KeyType* keys, size_t numKeys, const Box<T>& box)
{
    if constexpr (useGpu) { computeSfcKeysGpu(x, y, z, keys, numKeys, box); }
    else { computeSfcKeys(x, y, z, keys, numKeys, box); }
}

template<bool useGpu, class KeyType, class T>
void computeSfcMixDKeys(const T* x, const T* y, const T* z, KeyType* keys, size_t numKeys, const Box<T>& box, unsigned bx, unsigned by, unsigned bz)
{
    // if constexpr (useGpu) { computeSfcMixDKeysGpu(x, y, z, keys, numKeys, box, bx, by, bz); }
    // else { computeSfcMixDKeys(x, y, z, keys, numKeys, box, bx, by, bz); }
    // TODO(iomaganaris): for now only CPU version
    computeSfcMixDKeys(x, y, z, keys, numKeys, box, bx, by, bz);
}


} // namespace cstone
