/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Exposes gather functionality to reorder arrays with a map
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cassert>
#include <cstdint>
#include <span>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/util/pack_buffers.hpp"

namespace cstone
{

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

    template<class KeyType, class KeyBuf, class ValueBuf>
    void updateMap(std::span<KeyType> keys, KeyBuf& keyBuf, ValueBuf& valueBuf)
    {
        sortByKeyGpu<KeyType, IndexType>(keys, {ordering(), keys.size()}, keyBuf, valueBuf, growthRate_);
    }

    /*! @brief sort given Morton codes on the device and determine reorder map based on sort order
     */
    template<class KeyType, class KeyBuf, class ValueBuf>
    void setMapFromCodes(std::span<KeyType> keys, IndexType /*offset*/, KeyBuf& keyBuf, ValueBuf& valueBuf)
    {
        mapSize_ = keys.size();
        reallocateBytes(buffer_, keys.size() * sizeof(IndexType), growthRate_);
        sequenceGpu(ordering(), keys.size(), IndexType(0));
        sortByKeyGpu(keys, std::span<IndexType>(ordering(), keys.size()), keyBuf, valueBuf, growthRate_);
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
    IndexType* ordering() { return reinterpret_cast<IndexType*>(buffer_.data()); }
    const IndexType* ordering() const { return reinterpret_cast<const IndexType*>(buffer_.data()); }

    //! @brief reference to (non-owning) buffer for ordering
    BufferType& buffer_;
    IndexType mapSize_{0};
    float growthRate_ = 1.05;
};

} // namespace cstone
