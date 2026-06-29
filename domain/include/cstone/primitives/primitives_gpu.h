/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Basic algorithms on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <span>
#include <tuple>

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/cuda/device_vector.h"
#include "cstone/execution.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/util/pack_buffers.hpp"

namespace cstone
{

template<class T>
extern void fill(execution::Gpu exec, T* first, T* last, T value);

template<class T1, class T2>
inline void fill(execution::Gpu exec, T1* first, T1* last, T2 value)
{
    fill(exec, first, last, T1(value));
}

template<class T>
void copy_n(execution::Gpu exec, const T* src, std::size_t n, T* dest)
{
    memcpyD2DAsync(exec, src, n, dest);
}

template<class T1, class T2, class T3>
void scale(execution::Gpu exec, const T1* in1, const T1* in2, T2* out, T3 value);

template<class TS, class TD, class IndexType>
extern void gather(execution::Gpu exec, const IndexType* ordering, size_t numElements, const TS* src, TD* buffer);

template<class IndexType, class ValueType>
inline void
gather(execution::Gpu exec, std::span<const IndexType> ordering, const ValueType* source, ValueType* destination)
{
    gather(exec, ordering.data(), ordering.size(), source, destination);
}

template<class T, class IndexType>
extern void scatter(execution::Gpu exec, const IndexType* ordering, size_t numElements, const T* src, T* buffer);

template<class IndexType, class ValueType>
inline void
scatter(execution::Gpu exec, std::span<const IndexType> ordering, const ValueType* source, ValueType* destination)
{
    scatter(exec, ordering.data(), ordering.size(), source, destination);
}

template<class T, class IndexType>
extern void gatherScatter(
    execution::Gpu exec, const IndexType* gmap, const IndexType* smap, size_t numElements, const T* src, T* buffer);

template<class T>
extern std::tuple<T, T> minMax(execution::Gpu exec, const T* first, const T* last);

template<class T>
T maxNormSquare(execution::Gpu exec, const T* x, const T* y, const T* z, size_t numElements);

template<class T>
extern size_t lowerBound(execution::Gpu exec, const T* first, const T* last, T value);

template<class T, class IndexType>
extern void lowerBound(
    execution::Gpu exec, const T* first, const T* last, const T* valueFirst, const T* valueLast, IndexType* result);

template<class T1, class T2, class Tout>
extern void sequenceMax(execution::Gpu exec, const T1* i1_begin, const T1* i1_end, const T2* i2, Tout* output);

template<class Tin, class Tout>
extern Tout reduce(execution::Gpu exec, const Tin* input, size_t numElements, Tout init);

template<class IndexType>
extern void sequence(execution::Gpu exec, IndexType* input, size_t numElements, IndexType init);

template<class T1, class T2>
inline void sequence(execution::Gpu exec, T1* first, T1* last, T2 value)
{
    sequence(exec, first, last - first, T1(value));
}

template<class BufferType>
inline void sequence(execution::Gpu exec, LocalIndex first, LocalIndex n, BufferType& buffer, double growthRate)
{
    reallocateBytes(buffer, sizeof(LocalIndex) * (first + n), growthRate);
    auto* seq = reinterpret_cast<LocalIndex*>(buffer.data());
    sequence(exec, seq + first, seq + first + n, first);
}

/*! @brief sort range [first:last], using @p keyBuf as temporary storage
 *
 * @param[in]    exec     execution policy
 * @param[inout] first    pointer to first element in range
 * @param[inout] last     pointer to last element in range
 * @param[-]     keyBuf   buffer of length last-first for temporary usage
 */
template<class KeyType>
extern void sort(execution::Gpu exec, KeyType* first, KeyType* last, KeyType* keyBuf);

//! @brief Determine temporary device storage requirements for sortByKey
template<class KeyType, class ValueType>
extern uint64_t sortByKeyTempStorage(uint64_t numElements);

template<class KeyType, class ValueType>
extern void sortByKey(execution::Gpu exec,
                      KeyType* first,
                      KeyType* last,
                      ValueType* values,
                      KeyType* keyBuf,
                      ValueType* valueBuf,
                      void*,
                      uint64_t);

//! @brief sortByKey with temp buffer management
template<class KeyType, class ValueType, class KeyBuf, class ValueBuf>
inline void sortByKey(execution::Gpu exec,
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

    sortByKey(exec, keys.data(), keys.data() + keys.size(), values.data(), (KeyType*)rawPtr(keyBuf),
              tempBuffers[0].data(), tempBuffers[1].data(), tempStorageEle * sizeof(ValueType));
    syncGpu(exec);
    reallocate(keyBuf, s1, 1.0);
    reallocate(valueBuf, s2, 1.0);
}

template<class IndexType, class SumType>
extern void
exclusiveScan(execution::Gpu exec, const IndexType* first, const IndexType* last, SumType* output, SumType init);

template<class IndexType, class SumType>
extern void inclusiveScan(execution::Gpu exec, const IndexType* first, const IndexType* last, SumType* output);

template<class IndexType, class SumType>
inline void exclusiveScan(execution::Gpu exec, const IndexType* first, const IndexType* last, SumType* output)
{
    exclusiveScan(exec, first, last, output, SumType(0));
}

template<class ValueType>
extern size_t count(execution::Gpu exec, const ValueType* first, const ValueType* last, ValueType v);

template<class TS, class TD, class S>
extern void selectCopy(execution::Gpu exec, const TS* src, LocalIndex n, const S* selectFlags, TD* dest);

} // namespace cstone
