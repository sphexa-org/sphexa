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
#include <type_traits>

#include "cstone/cuda/device_vector.h"
#include "cstone/util/pack_buffers.hpp"
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

template<class It, class T>
void fill(execution::Gpu exec, It first, It last, T value)
{
    if (last <= first) { return; }
    using T1 = std::decay_t<decltype(*first)>;
    fillGpu(exec, first, last, T1(value));
}

template<class T>
void copy_n(execution::Cpu, const T* src, std::size_t n, T* dest)
{
    omp_copy(src, src + n, dest);
}

template<class T>
void copy_n(execution::Gpu exec, const T* src, std::size_t n, T* dest)
{
    memcpyD2DAsync(src, n, dest, exec);
}

template<class T1, class T2, class T3>
void scaleGpuAcc(execution::Cpu, const T1* in1, const T1* in2, T2* out, T3 value)
{
    std::transform(in1, in2, out, [value](auto v_) { return v_ * value; });
}

template<class T1, class T2, class T3>
void scaleGpuAcc(execution::Gpu exec, const T1* in1, const T1* in2, T2* out, T3 value)
{
    scaleGpu(exec, in1, in2, out, value);
}

template<class IndexType, class ValueType>
void gatherAcc(execution::Cpu, std::span<const IndexType> ordering, const ValueType* source, ValueType* destination)
{
    gather(ordering, source, destination);
}

template<class IndexType, class ValueType>
void gatherAcc(execution::Gpu exec,
               std::span<const IndexType> ordering,
               const ValueType* source,
               ValueType* destination)
{
    gatherGpu(exec, ordering.data(), ordering.size(), source, destination);
}

template<class IndexType, class ValueType>
void scatterAcc(execution::Cpu, std::span<const IndexType> ordering, const ValueType* source, ValueType* destination)
{
    scatter(ordering, source, destination);
}

template<class IndexType, class ValueType>
void scatterAcc(execution::Gpu exec,
                std::span<const IndexType> ordering,
                const ValueType* source,
                ValueType* destination)
{
    scatterGpu(exec, ordering.data(), ordering.size(), source, destination);
}

//! @brief sortByKey with temp buffer management
template<class KeyType, class ValueType, class KeyBuf, class ValueBuf>
void sortByKeyGpu(execution::Cpu,
                  std::span<KeyType> keys,
                  std::span<ValueType> values,
                  KeyBuf& /*keyBuf*/,
                  ValueBuf& /*valueBuf*/,
                  float /*growthRate*/)
{
    assert(keys.size() == values.size());
    sort_by_key(keys.begin(), keys.end(), values.begin());
}

//! @brief sortByKey with temp buffer management
template<class KeyType, class ValueType, class KeyBuf, class ValueBuf>
void sortByKeyGpu(execution::Gpu exec,
                  std::span<KeyType> keys,
                  std::span<ValueType> values,
                  KeyBuf& keyBuf,
                  ValueBuf& valueBuf,
                  float growthRate)
{
    // temp storage for radix sort as multiples of IndexType
    uint64_t tempStorageEle = iceil(sortByKeyTempStorage<KeyType, ValueType>(keys.size()), sizeof(ValueType));
    auto s1                 = reallocateBytes(keyBuf, keys.size() * sizeof(KeyType), growthRate);

    // pack valueBuffer and temp storage into @p valueBuf
    auto s2                 = valueBuf.size();
    uint64_t numElements[2] = {uint64_t(keys.size() * growthRate), tempStorageEle};
    auto tempBuffers        = util::packAllocBuffer<ValueType>(valueBuf, {numElements, 2}, 128);

    sortByKeyGpu(exec, keys.data(), keys.data() + keys.size(), values.data(), (KeyType*)rawPtr(keyBuf),
                 tempBuffers[0].data(), tempBuffers[1].data(), tempStorageEle * sizeof(ValueType));
    reallocate(keyBuf, s1, 1.0);
    reallocate(valueBuf, s2, 1.0);
}

template<class T1, class T2>
void sequenceAcc(execution::Cpu, T1* first, T1* last, T2 value)
{
    std::iota(first, last, value);
}

template<class T1, class T2>
void sequenceAcc(execution::Gpu exec, T1* first, T1* last, T2 value)
{
    sequenceGpu(exec, first, last - first, T1(value));
}

template<class BufferType>
void sequence(execution::Cpu exec, LocalIndex first, LocalIndex n, BufferType& buffer, double growthRate)
{
    reallocateBytes(buffer, sizeof(LocalIndex) * (first + n), growthRate);
    auto* seq = reinterpret_cast<LocalIndex*>(buffer.data());
    sequenceAcc(exec, seq + first, seq + first + n, first);
}

template<class BufferType>
void sequence(execution::Gpu exec, LocalIndex first, LocalIndex n, BufferType& buffer, double growthRate)
{
    reallocateBytes(buffer, sizeof(LocalIndex) * (first + n), growthRate);
    auto* seq = reinterpret_cast<LocalIndex*>(buffer.data());
    sequenceAcc(exec, seq + first, seq + first + n, first);
}

template<class KeyType, class ValueType>
void sortByKey(execution::Cpu, std::span<KeyType> keys, std::span<ValueType> values)
{
    assert(keys.size() == values.size());
    sort_by_key(keys.begin(), keys.end(), values.begin());
}

//! @brief CPU overload ignores scratch buffers and growth rate
template<class KeyType, class ValueType, class KeyBuf, class ValueBuf>
void sortByKey(execution::Cpu,
               std::span<KeyType> keys,
               std::span<ValueType> values,
               KeyBuf& /*keyBuf*/,
               ValueBuf& /*valueBuf*/,
               double /*growth*/)
{
    assert(keys.size() == values.size());
    sort_by_key(keys.begin(), keys.end(), values.begin());
}

template<class KeyType, class ValueType, class KeyBuf, class ValueBuf>
void sortByKey(execution::Gpu exec,
               std::span<KeyType> keys,
               std::span<ValueType> values,
               KeyBuf& keyBuf,
               ValueBuf& valueBuf,
               double growth)
{
    assert(keys.size() == values.size());
    sortByKeyGpu(exec, keys, values, keyBuf, valueBuf, growth);
}

} // namespace cstone
