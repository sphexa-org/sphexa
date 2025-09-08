/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Buffer description and management for domain decomposition particle exchanges
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/domain/index_ranges.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/util/pack_buffers.hpp"
#include "cstone/util/tuple_util.hpp"

namespace cstone
{
using util::computeByteOffsets;
using util::packBufferPtrs;

/*! @brief Layout description for particle buffers
 *
 * Common usage: storing the sub-range of locally owned/assigned particles within particle buffers
 *
 * 0       start              end      size
 * |-------|------------------|--------|
 *   halos   locally assigned   halos
 */
struct BufferDescription
{
    //! @brief subrange start
    LocalIndex start;
    //! @brief subrange end
    LocalIndex end;
    //! @brief total size of the buffer
    LocalIndex size;
};

template<int alignment, class... Arrays>
size_t computeTotalSendBytes(const SendList& sendList, int thisRank, size_t numBytesHeader, Arrays... arrays)
{
    size_t totalSendBytes = 0;
    for (int destinationRank = 0; destinationRank < int(sendList.size()); ++destinationRank)
    {
        size_t sendCount = sendList[destinationRank].totalCount();
        if (destinationRank == thisRank || sendCount == 0) { continue; }

        size_t particleBytes = util::computeByteOffsets(sendCount, alignment, arrays...).back();
        //! we add @a alignment bytes to the start of each message to provide space for the message length
        totalSendBytes += numBytesHeader + particleBytes;
    }

    return totalSendBytes;
}

template<int alignment, class... Arrays>
size_t computeTotalSendBytes(const SendRanges& sends, int thisRank, size_t numBytesHeader, Arrays... arrays)
{
    size_t totalSendBytes = 0;
    for (int destinationRank = 0; destinationRank < sends.numRanks(); ++destinationRank)
    {
        size_t sendCount = sends.count(destinationRank);
        if (destinationRank == thisRank || sendCount == 0) { continue; }

        size_t particleBytes = util::computeByteOffsets(sendCount, alignment, arrays...).back();
        //! we add @a alignment bytes to the start of each message to provide space for the message length
        totalSendBytes += numBytesHeader + particleBytes;
    }

    return totalSendBytes;
}

//! @brief Gather @p numElements of each array accessed through @p ordering into @p buffer. CPU and GPU.
template<int alignment, class F, class... Arrays>
std::size_t packArrays(F&& gather, const LocalIndex* ordering, LocalIndex numElements, char* buffer, Arrays... arrays)
{
    auto gatherArray = [&gather, numElements, ordering](auto arrayPair)
    { gather({ordering, numElements}, arrayPair[0], arrayPair[1]); };

    auto packTuple = util::packBufferPtrs<alignment>(buffer, numElements, arrays...);
    util::for_each_tuple(gatherArray, packTuple);

    std::size_t numBytesPacked = util::computeByteOffsets(numElements, alignment, arrays...).back();
    return numBytesPacked;
}

namespace domain_exchange
{
//! @brief return the required buffer size for calling exchangeParticles
[[maybe_unused]] static LocalIndex
exchangeBufferSize(BufferDescription bufDesc, LocalIndex numPresent, LocalIndex numAssigned)
{
    LocalIndex numIncoming = numAssigned - numPresent;

    bool fitHead = bufDesc.start >= numIncoming;
    bool fitTail = bufDesc.size - bufDesc.end >= numIncoming;

    return (fitHead || fitTail) ? bufDesc.size : bufDesc.end + numIncoming;
}

//! @brief return the index where particles from remote ranks will be received
[[maybe_unused]] static LocalIndex receiveStart(BufferDescription bufDesc, LocalIndex numIncoming)
{
    bool fitHead = bufDesc.start >= numIncoming;
    assert(fitHead || /*fitTail*/ bufDesc.size - bufDesc.end >= numIncoming);

    if (fitHead) { return bufDesc.start - numIncoming; }
    else { return bufDesc.end; }
}

//! @brief The index range that contains the locally assigned particles. Can contain left-over particles too.
[[maybe_unused]] static util::array<LocalIndex, 2> assignedEnvelope(BufferDescription bufDesc, LocalIndex numIncoming)
{
    bool fitHead = bufDesc.start >= numIncoming;
    if (fitHead) { return {bufDesc.start - numIncoming, bufDesc.end}; }
    else { return {bufDesc.start, bufDesc.end + numIncoming}; }
}

//! @brief realise o1 ordering with gather, then append received elements
template<class Vector>
void extractLocallyOwnedImpl(
    BufferDescription o1, LocalIndex numPresent, LocalIndex numAssigned, const LocalIndex* ordering, Vector& buffer)
{
    Vector temp(numAssigned);

    // extract what we already had before the exchange
    gatherCpu({ordering + o1.start, numPresent}, buffer.data(), temp.data());

    // extract what we received during the exchange
    LocalIndex rStart = receiveStart(o1, numAssigned - numPresent);
    std::copy_n(buffer.data() + rStart, numAssigned - numPresent, temp.data() + numPresent);
    swap(temp, buffer);
}

/*! @brief Only used in testing to isolate locally owned particles after calling exchangeParticles.
 *         In production code, this step is deferred until after halo detection to rearrange particles to their final
 *         location in a single step.
 */
template<class... Vector>
void extractLocallyOwned(BufferDescription bufDesc,
                         LocalIndex numPresent,
                         LocalIndex numAssigned,
                         const LocalIndex* ordering,
                         Vector&... buffers)
{
    std::initializer_list<int>{(extractLocallyOwnedImpl(bufDesc, numPresent, numAssigned, ordering, buffers), 0)...};
}

} // namespace domain_exchange
} // namespace cstone
