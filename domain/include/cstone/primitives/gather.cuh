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
    void updateMap(std::span<KeyType> keys, IndexType offset, KeyBuf& keyBuf, ValueBuf& valueBuf)
    {
        sortByKeyGpu<KeyType, IndexType>(keys, {ordering() + offset, keys.size()}, keyBuf, valueBuf, growthRate_);
    }

    //! @brief sort given SFC keys on the device and determine reorder map based on sort order
    template<class KeyType, class KeyBuf, class ValueBuf>
    void setMapFromCodes(std::span<KeyType> keys, IndexType offset, KeyBuf& keyBuf, ValueBuf& valueBuf)
    {
        reallocateBytes(buffer_, (keys.size() + offset) * sizeof(IndexType), growthRate_);
        sequenceGpu(ordering() + offset, keys.size(), offset);
        sortByKeyGpu(keys, std::span<IndexType>(ordering() + offset, keys.size()), keyBuf, valueBuf, growthRate_);
    }

    auto gatherFunc() const { return gatherGpuL; }

    //! @brief extend the ordering buffer to an additional range
    void extendMap(IndexType first, IndexType n) { sequenceGpu(ordering() + first, n, first); }

private:
    IndexType* ordering() { return reinterpret_cast<IndexType*>(buffer_.data()); }
    const IndexType* ordering() const { return reinterpret_cast<const IndexType*>(buffer_.data()); }

    //! @brief reference to (non-owning) buffer for ordering
    BufferType& buffer_;
    float growthRate_ = 1.05;
};

} // namespace cstone
