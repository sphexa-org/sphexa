/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Exchange particles between different ranks to satisfy their assignments of the global octree
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cstring>
#include <climits>

#include "buffer_description.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"

namespace cstone
{

//! @brief number of elements of all @p arrays that fit into @p numBytesAvail
template<size_t Alignment, class... Arrays>
size_t numElementsFit(size_t numBytesAvail, Arrays... arrays)
{
    constexpr int bytesPerElement = (... + sizeof(std::decay_t<decltype(*arrays)>));
    numBytesAvail -= sizeof...(arrays) * Alignment;
    return numBytesAvail / bytesPerElement;
}

inline void encodeSendCountCpu(uint64_t count, char* sendPtr) { memcpy(sendPtr, &count, sizeof(uint64_t)); }

template<class T>
uint64_t decodeSendCountCpu(T* recvPtr)
{
    uint64_t ret;
    memcpy(&ret, recvPtr, sizeof(uint64_t));
    return ret;
}

/*! @brief exchange array elements with other ranks according to the specified ranges
 *
 * @tparam Arrays             pointers to particles buffers
 * @param[in] epoch           MPI tag offset to avoid mix-ups of message from consecutive function calls
 * @param[in] receiveLog      List of received messages in previous calls to replicate resulting buffer layout
 * @param[in] sends           List of index ranges to be sent to each rank, indices
 *                            are valid w.r.t to arrays present on @p thisRank relative to @p particleStart.
 * @param[in] thisRank        Rank of the executing process
 * @param[in] receiveStart    start of receive index range where incoming particles in @p arrays will be placed
 * @param[in] receiveEnd      end of receive range
 * @param[in] ordering        Ordering through which to access arrays, valid w.r.t to [particleStart:particleEnd]
 * @param[inout] arrays       Pointers of different types but identical number of elements. The index range based
 *                            exchange operations performed are identical for each input array. Upon completion,
 *                            arrays will contain elements from the specified ranges and ranks.
 *                            The order in which the incoming ranges are grouped is random.
 *
 *  Example: If sendList[ri] contains the range [upper, lower), all elements (arrays+inputOffset)[ordering[upper:lower]]
 *           will be sent to rank ri. At the destination ri, the incoming elements
 *           will be either prepended or appended to elements already present on ri.
 *           No information about incoming particles to @p thisRank is contained in the function arguments,
 *           only their total number @p nParticlesAssigned, which also includes any assigned particles
 *           already present on @p thisRank.
 */
template<class... Arrays>
void exchangeParticles(int epoch,
                       ExchangeLog& receiveLog,
                       const SendRanges& sends,
                       int thisRank,
                       LocalIndex receiveStart,
                       LocalIndex receiveEnd,
                       const LocalIndex* ordering,
                       Arrays... arrays)
{
    using TransferType        = uint64_t;
    constexpr int alignment   = sizeof(TransferType);
    constexpr int headerBytes = round_up(sizeof(uint64_t), alignment);
    bool record               = receiveLog.empty();
    int domExTag              = static_cast<int>(P2pTags::domainExchange) + epoch;

    std::vector<std::vector<char>> sendBuffers;
    std::vector<MPI_Request> sendRequests;

    for (int destinationRank = 0; destinationRank < sends.numRanks(); ++destinationRank)
    {
        size_t sendCount = sends.count(destinationRank);
        if (destinationRank == thisRank || sendCount == 0) { continue; }

        size_t numSent = 0;
        while (numSent < sendCount)
        {
            size_t numRemaining  = sendCount - numSent;
            size_t numFit        = numElementsFit<alignment>(INT_MAX * sizeof(TransferType) - headerBytes, arrays...);
            size_t nextSendCount = std::min(numFit, numRemaining);

            std::vector<char> sendBuffer(headerBytes +
                                         util::computeByteOffsets(nextSendCount, alignment, arrays...).back());
            encodeSendCountCpu(nextSendCount, sendBuffer.data());
            packArrays<alignment>(gatherCpu, ordering + sends[destinationRank] + numSent, nextSendCount,
                                  sendBuffer.data() + headerBytes, arrays...);

            mpiSendAsyncAs<TransferType>(sendBuffer.data(), sendBuffer.size(), destinationRank, domExTag, sendRequests);
            numSent += nextSendCount;
            sendBuffers.push_back(std::move(sendBuffer));
        }
        assert(numSent == sendCount);
    }

    std::vector<char> receiveBuffer;
    while (receiveStart != receiveEnd)
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, domExTag, MPI_COMM_WORLD, &status);
        int receiveRank = status.MPI_SOURCE;
        int receiveCountTransfer;
        MPI_Get_count(&status, MpiType<TransferType>{}, &receiveCountTransfer);

        receiveBuffer.resize(receiveCountTransfer * sizeof(TransferType));
        mpiRecvSyncAs<TransferType>(receiveBuffer.data(), receiveBuffer.size(), receiveRank, domExTag, &status);

        size_t receiveCount = decodeSendCountCpu(receiveBuffer.data());
        assert(receiveStart + receiveCount <= receiveEnd);

        LocalIndex receiveLocation = receiveStart;
        if (record) { receiveLog.addExchange(receiveRank, receiveStart); }
        else { receiveLocation = receiveLog.lookup(receiveRank); }

        char* particleData = receiveBuffer.data() + headerBytes;
        auto packTuple     = util::packBufferPtrs<alignment>(particleData, receiveCount, (arrays + receiveLocation)...);
        auto scatterRanges = [receiveCount](auto arrayPair) { std::copy_n(arrayPair[1], receiveCount, arrayPair[0]); };
        util::for_each_tuple(scatterRanges, packTuple);

        receiveStart += receiveCount;
    }

    if (not sendRequests.empty())
    {
        MPI_Status status[sendRequests.size()];
        MPI_Waitall(int(sendRequests.size()), sendRequests.data(), status);
    }

    // If this process is going to send messages with rank/tag combinations
    // already sent in this function, this can lead to messages being mixed up
    // on the receiver side. This happens e.g. with repeated consecutive calls of
    // this function.

    // MUST call MPI_Barrier or any other collective MPI operation that enforces synchronization
    // across all ranks before calling this function again.
    // MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace cstone
