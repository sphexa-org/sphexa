/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  CPU/GPU wrappers for basic algorithms
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <type_traits>

#include "gather.hpp"
#include "cstone/primitives/primitives_gpu.h"

namespace cstone
{

struct CpuTag
{
};
struct GpuTag
{
};

template<class AccType>
struct HaveGpu : public std::integral_constant<int, std::is_same_v<AccType, GpuTag>>
{
};

template<bool useGpu, class T>
void fill(T* first, T* last, T value)
{
    if (last <= first) { return; }

    if constexpr (useGpu) { fillGpu(first, last, value); }
    else { std::fill(first, last, value); }
}

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
