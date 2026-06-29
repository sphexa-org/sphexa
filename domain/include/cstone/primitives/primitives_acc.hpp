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
#include <span>

#include "cstone/execution.hpp"
#include "cstone/primitives/primitives_gpu.h"
#include "gather.hpp"

namespace cstone
{

template<class It, class T>
void fill(execution::Cpu, It first, It last, T value)
{
    if (last <= first) { return; }
    std::fill(first, last, value);
}

template<class T>
void copy_n(execution::Cpu, const T* src, std::size_t n, T* dest)
{
    omp_copy(src, src + n, dest);
}

template<class T1, class T2, class T3>
void scale(execution::Cpu, const T1* in1, const T1* in2, T2* out, T3 value)
{
    std::transform(in1, in2, out, [value](auto v_) { return v_ * value; });
}

template<class IndexType, class ValueType>
void gather(execution::Cpu, std::span<const IndexType> ordering, const ValueType* source, ValueType* destination)
{
    gather(ordering, source, destination);
}

template<class IndexType, class ValueType>
void scatter(execution::Cpu, std::span<const IndexType> ordering, const ValueType* source, ValueType* destination)
{
    scatter(ordering, source, destination);
}

template<class T>
std::tuple<T, T> minMax(execution::Cpu, const T* first, const T* last)
{
    assert(last >= first);

    T minimum = INFINITY;
    T maximum = -INFINITY;

#pragma omp parallel for reduction(min : minimum) reduction(max : maximum)
    for (size_t pi = 0; pi < std::size_t(last - first); pi++)
    {
        T value = first[pi];
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }

    return std::make_tuple(minimum, maximum);
}

//! @brief sortByKey with temp buffer management
template<class KeyType, class ValueType, class KeyBuf, class ValueBuf>
void sortByKey(execution::Cpu,
               std::span<KeyType> keys,
               std::span<ValueType> values,
               KeyBuf& /*keyBuf*/,
               ValueBuf& /*valueBuf*/,
               float /*growthRate*/)
{
    assert(keys.size() == values.size());
    sort_by_key(keys.begin(), keys.end(), values.begin());
}

template<class T1, class T2>
void sequence(execution::Cpu, T1* first, T1* last, T2 value)
{
    std::iota(first, last, value);
}

template<class BufferType>
void sequence(execution::Cpu exec, LocalIndex first, LocalIndex n, BufferType& buffer, double growthRate)
{
    reallocateBytes(buffer, sizeof(LocalIndex) * (first + n), growthRate);
    auto* seq = reinterpret_cast<LocalIndex*>(buffer.data());
    sequence(exec, seq + first, seq + first + n, first);
}

} // namespace cstone
