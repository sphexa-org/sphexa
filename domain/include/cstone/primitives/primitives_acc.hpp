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
void fill(It first, It last, T value, execution::Cpu)
{
    if (last <= first) { return; }
    std::fill(first, last, value);
}

template<class It, class T>
void fill(It first, It last, T value, execution::Gpu stream)
{
    if (last <= first) { return; }
    using T1 = std::decay_t<decltype(*first)>;
    fillGpu(first, last, T1(value), stream);
}

template<class T>
void copy_n(const T* src, std::size_t n, T* dest, execution::Cpu)
{
    omp_copy(src, src + n, dest);
}

template<class T>
void copy_n(const T* src, std::size_t n, T* dest, execution::Gpu stream)
{
    memcpyD2DAsync(src, n, dest, stream);
}

template<class T1, class T2, class T3>
void scaleGpuAcc(const T1* in1, const T1* in2, T2* out, T3 value, execution::Cpu)
{
    std::transform(in1, in2, out, [value](auto v_) { return v_ * value; });
}

template<class T1, class T2, class T3>
void scaleGpuAcc(const T1* in1, const T1* in2, T2* out, T3 value, execution::Gpu stream)
{
    scaleGpu(in1, in2, out, value, stream);
}

template<class IndexType, class ValueType>
void gatherAcc(std::span<const IndexType> ordering, const ValueType* source, ValueType* destination, execution::Cpu)
{
    gather(ordering, source, destination);
}

template<class IndexType, class ValueType>
void gatherAcc(std::span<const IndexType> ordering,
               const ValueType* source,
               ValueType* destination,
               execution::Gpu stream)
{
    gatherGpu(ordering.data(), ordering.size(), source, destination, stream);
}

template<class IndexType, class ValueType>
void scatterAcc(std::span<const IndexType> ordering, const ValueType* source, ValueType* destination, execution::Cpu)
{
    scatter(ordering, source, destination);
}

template<class IndexType, class ValueType>
void scatterAcc(std::span<const IndexType> ordering,
                const ValueType* source,
                ValueType* destination,
                execution::Gpu stream)
{
    scatterGpu(ordering.data(), ordering.size(), source, destination, stream);
}

//! @brief sortByKey with temp buffer management
template<class KeyType, class ValueType, class KeyBuf, class ValueBuf>
void sortByKeyGpu(std::span<KeyType> keys,
                  std::span<ValueType> values,
                  KeyBuf& /*keyBuf*/,
                  ValueBuf& /*valueBuf*/,
                  float /*growthRate*/,
                  execution::Cpu)
{
    assert(keys.size() == values.size());
    sort_by_key(keys.begin(), keys.end(), values.begin());
}

//! @brief sortByKey with temp buffer management
template<class KeyType, class ValueType, class KeyBuf, class ValueBuf>
void sortByKeyGpu(std::span<KeyType> keys,
                  std::span<ValueType> values,
                  KeyBuf& keyBuf,
                  ValueBuf& valueBuf,
                  float growthRate,
                  execution::Gpu stream)
{
    // temp storage for radix sort as multiples of IndexType
    uint64_t tempStorageEle = iceil(sortByKeyTempStorage<KeyType, ValueType>(keys.size()), sizeof(ValueType));
    auto s1                 = reallocateBytes(keyBuf, keys.size() * sizeof(KeyType), growthRate);

    // pack valueBuffer and temp storage into @p valueBuf
    auto s2                 = valueBuf.size();
    uint64_t numElements[2] = {uint64_t(keys.size() * growthRate), tempStorageEle};
    auto tempBuffers        = util::packAllocBuffer<ValueType>(valueBuf, {numElements, 2}, 128);

    sortByKeyGpu(keys.data(), keys.data() + keys.size(), values.data(), (KeyType*)rawPtr(keyBuf), tempBuffers[0].data(),
                 tempBuffers[1].data(), tempStorageEle * sizeof(ValueType), stream);
    reallocate(keyBuf, s1, 1.0);
    reallocate(valueBuf, s2, 1.0);
}

template<class T1, class T2>
void sequenceAcc(T1* first, T1* last, T2 value, execution::Cpu)
{
    std::iota(first, last, value);
}

template<class T1, class T2>
void sequenceAcc(T1* first, T1* last, T2 value, execution::Gpu stream)
{
    sequenceGpu(first, last - first, T1(value), stream);
}

template<class BufferType>
void sequence(LocalIndex first, LocalIndex n, BufferType& buffer, double growthRate, execution::Cpu)
{
    reallocateBytes(buffer, sizeof(LocalIndex) * (first + n), growthRate);
    auto* seq = reinterpret_cast<LocalIndex*>(buffer.data());
    sequenceAcc(seq + first, seq + first + n, first, execution::Cpu{});
}

template<class BufferType>
void sequence(LocalIndex first, LocalIndex n, BufferType& buffer, double growthRate, execution::Gpu stream)
{
    reallocateBytes(buffer, sizeof(LocalIndex) * (first + n), growthRate);
    auto* seq = reinterpret_cast<LocalIndex*>(buffer.data());
    sequenceAcc(seq + first, seq + first + n, first, stream);
}

template<class KeyType, class ValueType>
void sortByKey(std::span<KeyType> keys, std::span<ValueType> values, execution::Cpu)
{
    assert(keys.size() == values.size());
    sort_by_key(keys.begin(), keys.end(), values.begin());
}

//! @brief CPU overload ignores scratch buffers and growth rate
template<class KeyType, class ValueType, class KeyBuf, class ValueBuf>
void sortByKey(std::span<KeyType> keys,
               std::span<ValueType> values,
               KeyBuf& /*keyBuf*/,
               ValueBuf& /*valueBuf*/,
               double /*growth*/,
               execution::Cpu)
{
    assert(keys.size() == values.size());
    sort_by_key(keys.begin(), keys.end(), values.begin());
}

template<class KeyType, class ValueType, class KeyBuf, class ValueBuf>
void sortByKey(std::span<KeyType> keys,
               std::span<ValueType> values,
               KeyBuf& keyBuf,
               ValueBuf& valueBuf,
               double growth,
               execution::Gpu stream)
{
    assert(keys.size() == values.size());
    sortByKeyGpu(keys, values, keyBuf, valueBuf, growth, stream);
}

} // namespace cstone
