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
extern void fillGpu(execution::Gpu exec, T* first, T* last, T value);

template<class T1, class T2, class T3>
void scaleGpu(execution::Gpu exec, const T1* in1, const T1* in2, T2* out, T3 value);

template<class TS, class TD, class IndexType>
extern void gatherGpu(execution::Gpu exec, const IndexType* ordering, size_t numElements, const TS* src, TD* buffer);

//! @brief Lambda to avoid templated functors that would become template-template parameters when passed to functions.
inline auto gatherGpuL = [](std::span<const LocalIndex> ordering, const auto* src, auto* dest)
{
    // TODO: remove usage of default exec
    gatherGpu(0, ordering.data(), ordering.size(), src, dest);
};

template<class T, class IndexType>
extern void scatterGpu(execution::Gpu exec, const IndexType* ordering, size_t numElements, const T* src, T* buffer);

template<class T, class IndexType>
extern void gatherScatterGpu(
    execution::Gpu exec, const IndexType* gmap, const IndexType* smap, size_t numElements, const T* src, T* buffer);

template<class Execution, class T>
struct MinMax;

template<class T>
struct MinMax<execution::Gpu, T>
{
    std::tuple<T, T> operator()(const T* first, const T* last);

    execution::Gpu exec;
};

template<class T>
T maxNormSquareGpu(execution::Gpu exec, const T* x, const T* y, const T* z, size_t numElements);

template<class T>
extern size_t lowerBoundGpu(execution::Gpu exec, const T* first, const T* last, T value);

template<class T, class IndexType>
extern void lowerBoundGpu(
    execution::Gpu exec, const T* first, const T* last, const T* valueFirst, const T* valueLast, IndexType* result);

template<class T1, class T2, class Tout>
extern void sequenceMax(execution::Gpu exec, const T1* i1_begin, const T1* i1_end, const T2* i2, Tout* output);

template<class Tin, class Tout>
extern Tout reduceGpu(execution::Gpu exec, const Tin* input, size_t numElements, Tout init);

template<class IndexType>
extern void sequenceGpu(execution::Gpu exec, IndexType* input, size_t numElements, IndexType init);

/*! @brief sort range [first:last], using @p keyBuf as temporary storage
 *
 * @param[inout] first   pointer to first element in range
 * @param[inout] last    pointer to last element in range
 * @param[-]     keyBuf  buffer of length last-first for temporary usage
 */
template<class KeyType>
extern void sortGpu(execution::Gpu exec, KeyType* first, KeyType* last, KeyType* keyBuf);

//! @brief Determine temporary device storage requirements for sortByKeyGpu
template<class KeyType, class ValueType>
extern uint64_t sortByKeyTempStorage(uint64_t numElements);

template<class KeyType, class ValueType>
extern void sortByKeyGpu(execution::Gpu exec,
                         KeyType* first,
                         KeyType* last,
                         ValueType* values,
                         KeyType* keyBuf,
                         ValueType* valueBuf,
                         void*,
                         uint64_t);

template<class KeyType, class ValueType>
extern void sortByKeyGpu(execution::Gpu exec, KeyType* first, KeyType* last, ValueType* values);

template<class IndexType, class SumType>
extern void
exclusiveScanGpu(execution::Gpu exec, const IndexType* first, const IndexType* last, SumType* output, SumType init);

template<class IndexType, class SumType>
extern void inclusiveScanGpu(execution::Gpu exec, const IndexType* first, const IndexType* last, SumType* output);

template<class IndexType, class SumType>
void exclusiveScanGpu(execution::Gpu exec, const IndexType* first, const IndexType* last, SumType* output)
{
    exclusiveScanGpu(exec, first, last, output, SumType(0));
}

template<class ValueType>
extern size_t countGpu(execution::Gpu exec, const ValueType* first, const ValueType* last, ValueType v);

template<class TS, class TD, class S>
extern void selectCopyGpu(execution::Gpu exec, const TS* src, LocalIndex n, const S* selectFlags, TD* dest);

} // namespace cstone
