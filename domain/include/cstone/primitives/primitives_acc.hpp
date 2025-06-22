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

template<bool useGpu, class T>
void copy_n(const T* src, std::size_t n, T* dest)
{
    if constexpr (useGpu) { memcpyD2D(src, n, dest); }
    else { omp_copy(src, src + n, dest); }
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

//! @brief sortByKey with temp buffer management
template<class KeyType, class ValueType, class KeyBuf, class ValueBuf>
void sortByKeyGpu(
    std::span<KeyType> keys, std::span<ValueType> values, KeyBuf& keyBuf, ValueBuf& valueBuf, float growthRate)
{
    // temp storage for radix sort as multiples of IndexType
    uint64_t tempStorageEle = iceil(sortByKeyTempStorage<KeyType, ValueType>(keys.size()), sizeof(ValueType));
    auto s1                 = reallocateBytes(keyBuf, keys.size() * sizeof(KeyType), growthRate);

    // pack valueBuffer and temp storage into @p valueBuf
    auto s2                 = valueBuf.size();
    uint64_t numElements[2] = {uint64_t(keys.size() * growthRate), tempStorageEle};
    auto tempBuffers        = util::packAllocBuffer<ValueType>(valueBuf, {numElements, 2}, 128);

    sortByKeyGpu(keys.data(), keys.data() + keys.size(), values.data(), (KeyType*)rawPtr(keyBuf), tempBuffers[0].data(),
                 tempBuffers[1].data(), tempStorageEle * sizeof(ValueType));
    reallocate(keyBuf, s1, 1.0);
    reallocate(valueBuf, s2, 1.0);
}

template<bool useGpu, class BufferType>
void sequence(LocalIndex first, LocalIndex n, BufferType& buffer, double growthRate)
{
    reallocateBytes(buffer, sizeof(LocalIndex) * (first + n), growthRate);
    auto* seq = reinterpret_cast<LocalIndex*>(buffer.data());
    if constexpr (useGpu) { sequenceGpu(seq + first, n, first); }
    else { std::iota(seq + first, seq + first + n, first); }
}

template<bool useGpu, class KeyType, class ValueType, class KeyBuf, class ValueBuf>
void sortByKey(std::span<KeyType> keys, std::span<ValueType> values, KeyBuf& keyBuf, ValueBuf& valueBuf, double growth)
{
    assert(keys.size() == values.size());
    if constexpr (useGpu) { sortByKeyGpu(keys, values, keyBuf, valueBuf, growth); }
    else { sort_by_key(keys.begin(), keys.end(), values.begin()); }
}

} // namespace cstone
