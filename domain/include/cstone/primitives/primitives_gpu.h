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

#include "cstone/execution.hpp"
#include "cstone/tree/definitions.h"

namespace cstone
{

template<class T>
extern void fillGpu(T* first, T* last, T value, execution::Gpu exec);

template<class T1, class T2, class T3>
void scaleGpu(const T1* in1, const T1* in2, T2* out, T3 value, execution::Gpu exec);

template<class TS, class TD, class IndexType>
extern void gatherGpu(const IndexType* ordering, size_t numElements, const TS* src, TD* buffer, execution::Gpu exec);

//! @brief Lambda to avoid templated functors that would become template-template parameters when passed to functions.
inline auto gatherGpuL = [](std::span<const LocalIndex> ordering, const auto* src, auto* dest)
{
    // TODO: remove usage of default exec
    gatherGpu(ordering.data(), ordering.size(), src, dest, 0);
};

template<class T, class IndexType>
extern void scatterGpu(const IndexType* ordering, size_t numElements, const T* src, T* buffer, execution::Gpu exec);

template<class T, class IndexType>
extern void gatherScatterGpu(
    const IndexType* gmap, const IndexType* smap, size_t numElements, const T* src, T* buffer, execution::Gpu exec);

template<class Execution, class T>
struct MinMax;

template<class T>
struct MinMax<execution::Gpu, T>
{
    std::tuple<T, T> operator()(const T* first, const T* last);

    execution::Gpu exec;
};

template<class T>
T maxNormSquareGpu(const T* x, const T* y, const T* z, size_t numElements, execution::Gpu exec);

template<class T>
extern size_t lowerBoundGpu(const T* first, const T* last, T value, execution::Gpu exec);

template<class T, class IndexType>
extern void lowerBoundGpu(
    const T* first, const T* last, const T* valueFirst, const T* valueLast, IndexType* result, execution::Gpu exec);

template<class T1, class T2, class Tout>
extern void sequenceMax(const T1* i1_begin, const T1* i1_end, const T2* i2, Tout* output, execution::Gpu exec);

template<class Tin, class Tout>
extern Tout reduceGpu(const Tin* input, size_t numElements, Tout init, execution::Gpu exec);

template<class IndexType>
extern void sequenceGpu(IndexType* input, size_t numElements, IndexType init, execution::Gpu exec);

/*! @brief sort range [first:last], using @p keyBuf as temporary storage
 *
 * @param[inout] first   pointer to first element in range
 * @param[inout] last    pointer to last element in range
 * @param[-]     keyBuf  buffer of length last-first for temporary usage
 */
template<class KeyType>
extern void sortGpu(KeyType* first, KeyType* last, KeyType* keyBuf, execution::Gpu exec);

//! @brief Determine temporary device storage requirements for sortByKeyGpu
template<class KeyType, class ValueType>
extern uint64_t sortByKeyTempStorage(uint64_t numElements);

template<class KeyType, class ValueType>
extern void sortByKeyGpu(KeyType* first,
                         KeyType* last,
                         ValueType* values,
                         KeyType* keyBuf,
                         ValueType* valueBuf,
                         void*,
                         uint64_t,
                         execution::Gpu exec);

template<class KeyType, class ValueType>
extern void sortByKeyGpu(KeyType* first, KeyType* last, ValueType* values, execution::Gpu exec);

template<class IndexType, class SumType>
extern void
exclusiveScanGpu(const IndexType* first, const IndexType* last, SumType* output, SumType init, execution::Gpu exec);

template<class IndexType, class SumType>
extern void inclusiveScanGpu(const IndexType* first, const IndexType* last, SumType* output, execution::Gpu exec);

template<class IndexType, class SumType>
void exclusiveScanGpu(const IndexType* first, const IndexType* last, SumType* output, execution::Gpu exec)
{
    exclusiveScanGpu(first, last, output, SumType(0), exec);
}

template<class ValueType>
extern size_t countGpu(const ValueType* first, const ValueType* last, ValueType v, execution::Gpu exec);

template<class TS, class TD, class S>
extern void selectCopyGpu(const TS* src, LocalIndex n, const S* selectFlags, TD* dest, execution::Gpu exec);

} // namespace cstone
