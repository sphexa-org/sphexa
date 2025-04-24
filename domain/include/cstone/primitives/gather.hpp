/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Functionality for calculating for performing gather operations on the CPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <span>
#include <tuple>
#include <vector>

#include "cstone/tree/definitions.h"
#include "cstone/util/noinit_alloc.hpp"
#include "cstone/util/reallocate.hpp"

namespace cstone
{

/*! @brief sort values according to a key
 *
 * @param[inout] keyBegin    key sequence start
 * @param[inout] keyEnd      key sequence end
 * @param[inout] valueBegin  values
 * @param[in]    compare     comparison function
 *
 * Upon completion of this routine, the key sequence will be sorted and values
 * will be rearranged to reflect the key ordering
 */
template<class InoutIterator, class OutputIterator, class Compare>
void sort_by_key(InoutIterator keyBegin, InoutIterator keyEnd, OutputIterator valueBegin, Compare compare)
{
    using KeyType   = std::decay_t<decltype(*keyBegin)>;
    using ValueType = std::decay_t<decltype(*valueBegin)>;
    std::size_t n   = std::distance(keyBegin, keyEnd);

    // zip the input integer array together with the index sequence
    std::vector<std::tuple<KeyType, ValueType>, util::DefaultInitAdaptor<std::tuple<KeyType, ValueType>>> keyIndexPairs(
        n);
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i)
        keyIndexPairs[i] = std::make_tuple(keyBegin[i], valueBegin[i]);

    // sort, comparing only the first tuple element
    std::stable_sort(begin(keyIndexPairs), end(keyIndexPairs),
                     [compare](const auto& t1, const auto& t2) { return compare(std::get<0>(t1), std::get<0>(t2)); });

// extract the resulting ordering and store back the sorted keys
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i)
    {
        keyBegin[i]   = std::get<0>(keyIndexPairs[i]);
        valueBegin[i] = std::get<1>(keyIndexPairs[i]);
    }
}

//! @brief calculate the sortKey that sorts the input sequence, default ascending order
template<class InoutIterator, class OutputIterator>
void sort_by_key(InoutIterator inBegin, InoutIterator inEnd, OutputIterator outBegin)
{
    sort_by_key(inBegin, inEnd, outBegin, std::less<std::decay_t<decltype(*inBegin)>>{});
}

//! @brief copy with multiple OpenMP threads
template<class InputIterator, class OutputIterator>
void omp_copy(InputIterator first, InputIterator last, OutputIterator out)
{
    std::size_t n = last - first;

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i)
    {
        out[i] = first[i];
    }
}

//! @brief gather reorder
template<class IndexType, class ValueType>
void gather(std::span<const IndexType> ordering, const ValueType* source, ValueType* destination)
{
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < ordering.size(); ++i)
    {
        destination[i] = source[ordering[i]];
    }
}

//! @brief Lambda to avoid templated functors that would become template-template parameters when passed to functions.
inline auto gatherCpu = [](std::span<const LocalIndex> ordering, const auto* src, auto* dest)
{ gather<LocalIndex>(ordering, src, dest); };

//! @brief scatter reorder
template<class IndexType, class ValueType>
void scatter(std::span<const IndexType> ordering, const ValueType* source, ValueType* destination)
{
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < ordering.size(); ++i)
    {
        destination[ordering[i]] = source[i];
    }
}

//! @brief gather from @p src and scatter into @p dst
template<class IndexType, class VType>
void gatherScatter(std::span<const IndexType> gmap, std::span<const IndexType> smap, const VType* src, VType* dst)
{
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < gmap.size(); ++i)
    {
        dst[smap[i]] = src[gmap[i]];
    }
}

template<class BufferType>
class SfcSorter
{
public:
    SfcSorter(BufferType& buffer)
        : buffer_(buffer)
    {
    }

    SfcSorter(const SfcSorter&) = delete;

    LocalIndex* getMap() { return ordering(); }
    BufferType& getBuf() { return buffer_; }

private:
    LocalIndex* ordering() { return reinterpret_cast<LocalIndex*>(buffer_.data()); }
    const LocalIndex* ordering() const { return reinterpret_cast<const LocalIndex*>(buffer_.data()); }

    //! @brief reference to (non-owning) buffer for ordering
    BufferType& buffer_;
};

} // namespace cstone
