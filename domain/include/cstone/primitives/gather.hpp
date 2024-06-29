/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
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
#include <tuple>
#include <vector>

#include "cstone/tree/definitions.h"
#include "cstone/util/gsl-lite.hpp"
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
void gather(gsl::span<const IndexType> ordering, const ValueType* source, ValueType* destination)
{
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < ordering.size(); ++i)
    {
        destination[i] = source[ordering[i]];
    }
}

//! @brief Lambda to avoid templated functors that would become template-template parameters when passed to functions.
inline auto gatherCpu = [](const auto* ordering, auto numElements, const auto* src, const auto dest) {
    gather<LocalIndex>({ordering, numElements}, src, dest);
};

//! @brief scatter reorder
template<class IndexType, class ValueType>
void scatter(gsl::span<const IndexType> ordering, const ValueType* source, ValueType* destination)
{
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < ordering.size(); ++i)
    {
        destination[ordering[i]] = source[i];
    }
}

//! @brief gather from @p src and scatter into @p dst
template<class IndexType, class VType>
void gatherScatter(gsl::span<const IndexType> gmap, gsl::span<const IndexType> smap, const VType* src, VType* dst)
{
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < gmap.size(); ++i)
    {
        dst[smap[i]] = src[gmap[i]];
    }
}

template<class IndexType, class BufferType>
class SfcSorter
{
public:
    SfcSorter(BufferType& buffer)
        : buffer_(buffer)
    {
    }

    SfcSorter(const SfcSorter&) = delete;

    const IndexType* getMap() const { return ordering(); }
    std::size_t size() const { return mapSize_; }

    template<class KeyType>
    void setMapFromCodes(KeyType* first, KeyType* last)
    {
        mapSize_ = std::size_t(last - first);
        reallocateBytes(buffer_, mapSize_ * sizeof(IndexType), 1.0);
        std::iota(ordering(), ordering() + mapSize_, 0);
        sort_by_key(first, last, ordering());
    }

    template<class KeyType>
    void updateMap(KeyType* first, KeyType* last)
    {
        assert(last - first == mapSize_);
        sort_by_key(first, last, ordering());
    }

    auto gatherFunc() const { return gatherCpu; }

    /*! @brief extend ordering map to the left or right
     *
     * @param[in] shifts    number of shifts
     * @param[-]  scratch   scratch space for temporary usage
     *
     * Negative shift values extends the ordering map to the left, positive value to the right
     * Examples: map = [1, 0, 3, 2] -> extendMap(-1) -> map = [0, 2, 1, 4, 3]
     *           map = [1, 0, 3, 2] -> extendMap(1) -> map = [1, 0, 3, 2, 4]
     *
     * This is used to extend the key-buffer passed to setMapFromCodes with additional keys, without
     * having to restore the original unsorted key-sequence.
     */
    template<class Vector>
    void extendMap(std::make_signed_t<IndexType> shifts, Vector& scratch)
    {
        if (shifts == 0) { return; }

        auto newMapSize = mapSize_ + std::abs(shifts);
        reallocateBytes(scratch, newMapSize * sizeof(IndexType), 1.0);
        auto* tempMap = reinterpret_cast<IndexType*>(scratch.data());

        if (shifts < 0)
        {
            std::iota(tempMap, tempMap - shifts, IndexType(0));
            std::transform(ordering(), ordering() + mapSize_, tempMap - shifts,
                           [shifts](auto x) { return x - shifts; });
        }
        else if (shifts > 0)
        {
            omp_copy(ordering(), ordering() + mapSize_, tempMap);
            std::iota(tempMap + mapSize_, tempMap + newMapSize, mapSize_);
        }
        reallocateBytes(buffer_, newMapSize * sizeof(IndexType), 1.0);
        omp_copy(tempMap, tempMap + newMapSize, ordering());
        mapSize_ = newMapSize;
    }

private:
    IndexType* ordering() { return reinterpret_cast<IndexType*>(buffer_.data()); }
    const IndexType* ordering() const { return reinterpret_cast<const IndexType*>(buffer_.data()); }

    //! @brief reference to (non-owning) buffer for ordering
    BufferType& buffer_;
    std::size_t mapSize_{0};
};

} // namespace cstone
