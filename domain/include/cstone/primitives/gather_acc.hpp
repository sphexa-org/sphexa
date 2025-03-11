/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief CPU/GPU wrappers
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/primitives/gather.hpp"
#include "cstone/primitives/primitives_gpu.h"

namespace cstone
{

template<bool useGpu, class IndexType, class ValueType>
void gatherAcc(std::span<const IndexType> ordering, const ValueType* source, ValueType* destination)
{
    if constexpr (useGpu) { gatherGpu(ordering.data(), ordering.size(), source, destination); }
    else { gather(ordering, source, destination); }
}

template<bool useGpu, class IndexType, class ValueType>
void scatterAcc(std::span<const IndexType> ordering, const ValueType* source, ValueType* destination)
{
    if constexpr (useGpu) { scatterGpu(ordering.data(), ordering.size(), source, destination); }
    else { scatter(ordering, source, destination); }
}

} // namespace cstone
