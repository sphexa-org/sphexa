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
#include <cstone/tree/definitions.h>

namespace cstone
{

template<class T>
extern void fillGpu(T* first, T* last, T value);

template<class T>
extern void scaleGpu(T* first, T* last, T value);

template<class T>
extern void incrementGpu(const T* first, const T* last, T* d_first, T value);

template<class T, class IndexType>
extern void gatherGpu(const IndexType* ordering, size_t numElements, const T* src, T* buffer);

//! @brief Lambda to avoid templated functors that would become template-template parameters when passed to functions.
inline auto gatherGpuL = [](std::span<const LocalIndex> ordering, const auto* src, auto* dest)
{ gatherGpu(ordering.data(), ordering.size(), src, dest); };

template<class T, class IndexType>
extern void scatterGpu(const IndexType* ordering, size_t numElements, const T* src, T* buffer);

template<class T>
struct MinMaxGpu
{
    std::tuple<T, T> operator()(const T* first, const T* last);
};

template<class T>
T maxNormSquareGpu(const T* x, const T* y, const T* z, size_t numElements);

template<class T>
extern size_t lowerBoundGpu(const T* first, const T* last, T value);

template<class T, class IndexType>
extern void lowerBoundGpu(const T* first, const T* last, const T* valueFirst, const T* valueLast, IndexType* result);

/*! @brief determine maximum elements in an array divided into multiple segments
 *
 * @tparam      Tin          some type that supports comparison
 * @tparam      Tout         some type that supports comparison
 * @tparam      IndexType    32- or 64-bit unsigned integer
 * @param[in]   input        an array of length @a segments[numSegments]
 * @param[in]   segments     an array of length @a numSegments + 1 describing the segmentation of @a input
 * @param[in]   numSegments  number of segments
 * @param[out]  output       maximum in each segment, length @a numSegments
 */
template<class Tin, class Tout, class IndexType>
extern void segmentMax(const Tin* input, const IndexType* segments, size_t numSegments, Tout* output);

template<class Tin, class Tout>
extern Tout reduceGpu(const Tin* input, size_t numElements, Tout init);

template<class IndexType>
extern void sequenceGpu(IndexType* input, size_t numElements, IndexType init);

/*! @brief sort range [first:last], using @p keyBuf as temporary storage
 *
 * @param[inout] first   pointer to first element in range
 * @param[inout] last    pointer to last element in range
 * @param[-]     keyBuf  buffer of length last-first for temporary usage
 */
template<class KeyType>
extern void sortGpu(KeyType* first, KeyType* last, KeyType* keyBuf);

//! @brief Determine temporary device storage requirements for sortByKeyGpu
template<class KeyType, class ValueType>
extern uint64_t sortByKeyTempStorage(uint64_t numElements);

template<class KeyType, class ValueType>
extern void
sortByKeyGpu(KeyType* first, KeyType* last, ValueType* values, KeyType* keyBuf, ValueType* valueBuf, void*, uint64_t);

template<class KeyType, class ValueType>
extern void sortByKeyGpu(KeyType* first, KeyType* last, ValueType* values);

template<class IndexType, class SumType>
extern void exclusiveScanGpu(const IndexType* first, const IndexType* last, SumType* output, SumType init);

template<class IndexType, class SumType>
extern void inclusiveScanGpu(const IndexType* first, const IndexType* last, SumType* output);

template<class IndexType, class SumType>
void exclusiveScanGpu(const IndexType* first, const IndexType* last, SumType* output)
{
    exclusiveScanGpu(first, last, output, SumType(0));
}

template<class ValueType>
extern size_t countGpu(const ValueType* first, const ValueType* last, ValueType v);

template<class T>
extern void selectCopyGpu(const T* src, LocalIndex n, const int* selectFlags, T* dest);

} // namespace cstone
