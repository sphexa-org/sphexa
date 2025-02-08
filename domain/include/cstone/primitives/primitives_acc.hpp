/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  CPU/GPU Wrapper of basic algorithms
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>

#include "cstone/primitives/primitives_gpu.h"

namespace cstone
{

template<bool useGpu, class T>
void fill(T* first, T* last, T value)
{
    if (last <= first) { return; }

    if constexpr (useGpu) { fillGpu(first, last, value); }
    else { std::fill(first, last, value); }
}

} // namespace cstone
