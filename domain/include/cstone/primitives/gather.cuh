/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
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
 * @brief  Exposes gather functionality to reorder arrays with a map
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cassert>
#include <cstdint>
#include <memory>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/util/pack_buffers.hpp"

namespace cstone
{

template<class IndexType, class BufferType>
class GpuSfcSorter
{
public:
    GpuSfcSorter(BufferType& buffer)
        : buffer_(buffer)
    {
    }

    GpuSfcSorter(const GpuSfcSorter&) = delete;

    const IndexType* getMap() const { return ordering(); }

    /*! @brief sort given Morton codes on the device and determine reorder map based on sort order
     *
     * @param[inout] first   pointer to first SFC code
     * @param[inout] last    pointer to last SFC code
     *
     * Precondition:
     *   - [first:last] is a continuous sequence of accessible elements
     *
     * Postcondition
     *   - [first:last] is sorted
     *   - subsequent calls to operator() apply a gather operation to the input sequence
     *     with the map obtained from sort_by_key with [first:last] as the keys
     *     and the identity permutation as the values
     */
    template<class KeyType>
    void setMapFromCodes(KeyType* first, KeyType* last)
    {
        mapSize_ = std::size_t(last - first);
        reallocateBytes(buffer_, mapSize_ * sizeof(IndexType), growthRate_);
        sequenceGpu(ordering(), mapSize_, IndexType(0));
        sortByKeyGpu(first, last, ordering());
    }

    template<class KeyType, class KeyBuf, class ValueBuf>
    void updateMap(KeyType* first, KeyType* last, KeyBuf& keyBuf, ValueBuf& valueBuf)
    {
        assert(last - first == mapSize_);
        // temp storage for radix sort as multiples of IndexType
        uint64_t tempStorageEle = iceil(sortByKeyTempStorage<KeyType, IndexType>(last - first), sizeof(IndexType));

        auto s1 = reallocateBytes(keyBuf, mapSize_ * sizeof(KeyType), growthRate_);

        // pack valueBuffer and temp storage into @p valueBuf
        auto s2                 = valueBuf.size();
        uint64_t numElements[2] = {uint64_t(mapSize_ * growthRate_), tempStorageEle};
        auto tempBuffers        = util::packAllocBuffer<IndexType>(valueBuf, {numElements, 2}, 128);

        sortByKeyGpu(first, last, ordering(), (KeyType*)rawPtr(keyBuf), tempBuffers[0].data(), tempBuffers[1].data(),
                     tempStorageEle * sizeof(IndexType));
        reallocate(keyBuf, s1, 1.0);
        reallocate(valueBuf, s2, 1.0);
    }

    template<class KeyType, class KeyBuf, class ValueBuf>
    void setMapFromCodes(KeyType* first, KeyType* last, KeyBuf& keyBuf, ValueBuf& valueBuf)
    {
        mapSize_ = std::size_t(last - first);
        reallocateBytes(buffer_, mapSize_ * sizeof(IndexType), growthRate_);
        sequenceGpu(ordering(), mapSize_, IndexType(0));

        updateMap(first, last, keyBuf, valueBuf);
    }

    auto gatherFunc() const { return gatherGpuL; }

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
        auto s1         = reallocateBytes(scratch, newMapSize * sizeof(IndexType), 1.0);
        auto* tempMap   = reinterpret_cast<IndexType*>(rawPtr(scratch));

        if (shifts < 0)
        {
            sequenceGpu(tempMap, IndexType(-shifts), IndexType(0));
            incrementGpu(ordering(), ordering() + mapSize_, tempMap - shifts, IndexType(-shifts));
        }
        else if (shifts > 0)
        {
            memcpyD2D(ordering(), mapSize_, tempMap);
            sequenceGpu(tempMap + mapSize_, IndexType(shifts), mapSize_);
        }
        reallocateBytes(buffer_, newMapSize * sizeof(IndexType), 1.0);
        memcpyD2D(tempMap, newMapSize, ordering());
        mapSize_ = newMapSize;
        reallocate(scratch, s1, 1.0);
    }

private:
    IndexType* ordering() { return reinterpret_cast<IndexType*>(rawPtr(buffer_)); }
    const IndexType* ordering() const { return reinterpret_cast<const IndexType*>(rawPtr(buffer_)); }

    //! @brief reference to (non-owning) buffer for ordering
    BufferType& buffer_;
    IndexType mapSize_{0};
    float growthRate_ = 1.05;
};

} // namespace cstone
