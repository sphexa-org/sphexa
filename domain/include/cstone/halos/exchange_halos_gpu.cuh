/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Halo particle exchange with MPI point-to-point communication
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <numeric>
#include <vector>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/errorcheck.cuh"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/primitives/mpi_cuda.cuh"
#include "cstone/domain/buffer_description.hpp"
#include "cstone/util/reallocate.hpp"
#include "cstone/util/tuple_util.hpp"

#include "gather_halos_gpu.h"

namespace cstone
{

template<class DevVec1, class DevVec2, class... Arrays>
void haloExchangeGpu(int epoch,
                     const RecvList& incomingHalos,
                     const SendList& outgoingHalos,
                     DevVec1& sendScratchBuffer,
                     DevVec2& receiveScratchBuffer,
                     Arrays... arrays)
{
    using TransferType          = uint64_t;
    constexpr int alignment     = sizeof(TransferType);
    using IndexType             = SendManifest::IndexType;
    const float allocGrowthRate = 1.05;

    int haloExchangeTag = static_cast<int>(P2pTags::haloExchange) + epoch;
    std::vector<MPI_Request> sendRequests;
    std::vector<std::vector<TransferType, util::DefaultInitAdaptor<TransferType>>> sendBuffers;

    const size_t oldSendSize = reallocateBytes(
        sendScratchBuffer, computeTotalSendBytes<alignment>(outgoingHalos, -1, 0, arrays...), allocGrowthRate);

    size_t numRanges = maxNumRanges(outgoingHalos);
    IndexType* d_range;
    checkGpuErrors(cudaMalloc((void**)&d_range, 2 * numRanges * sizeof(IndexType)));
    IndexType* d_rangeScan = d_range + numRanges;

    auto* sendPtr = reinterpret_cast<TransferType*>(rawPtr(sendScratchBuffer));
    for (std::size_t destinationRank = 0; destinationRank < outgoingHalos.size(); ++destinationRank)
    {
        const auto& outHalos = outgoingHalos[destinationRank];
        size_t sendCount     = outHalos.totalCount();
        if (sendCount == 0) continue;

        checkGpuErrors(
            cudaMemcpy(d_range, outHalos.offsets(), outHalos.nRanges() * sizeof(IndexType), cudaMemcpyHostToDevice));
        checkGpuErrors(
            cudaMemcpy(d_rangeScan, outHalos.scan(), outHalos.nRanges() * sizeof(IndexType), cudaMemcpyHostToDevice));

        auto gatherArray = [d_range, d_rangeScan, numRanges = outHalos.nRanges(), sendCount](auto arrayPtr)
        { gatherRanges(d_rangeScan, d_range, numRanges, arrayPtr[0], arrayPtr[1], sendCount); };

        for_each_tuple(gatherArray, util::packBufferPtrs<alignment>(sendPtr, sendCount, arrays...));
        checkGpuErrors(cudaDeviceSynchronize());

        size_t numElementsSend = util::computeByteOffsets(sendCount, alignment, arrays...).back() / alignment;
        mpiSendGpuDirect(sendPtr, numElementsSend, int(destinationRank), haloExchangeTag, sendRequests, sendBuffers);
        sendPtr += numElementsSend;
    }

    int numMessages           = 0;
    LocalIndex maxReceiveSize = 0;
    for (const auto& incomingHalo : incomingHalos)
    {
        numMessages += int(incomingHalo.count() > 0);
        maxReceiveSize = std::max(maxReceiveSize, incomingHalo.count());
    }
    size_t maxReceiveBytes = util::computeByteOffsets(maxReceiveSize, alignment, arrays...).back();

    const size_t oldRecvSize = reallocateBytes(receiveScratchBuffer, maxReceiveBytes, allocGrowthRate);
    auto* receiveBuffer      = reinterpret_cast<TransferType*>(rawPtr(receiveScratchBuffer));

    while (numMessages--)
    {
        MPI_Status status;
        mpiRecvGpuDirect(receiveBuffer, maxReceiveBytes / alignment, MPI_ANY_SOURCE, haloExchangeTag, &status);
        int receiveRank         = status.MPI_SOURCE;
        const auto& inHalos     = incomingHalos[receiveRank];
        LocalIndex receiveCount = inHalos.count();

        auto unpack = [start = inHalos.start(), receiveCount](auto arrayPair)
        { memcpyD2H(arrayPair[1], receiveCount, arrayPair[0] + start); };

        for_each_tuple(unpack, util::packBufferPtrs<alignment>(receiveBuffer, receiveCount, arrays...));
        checkGpuErrors(cudaDeviceSynchronize());
    }

    if (not sendRequests.empty())
    {
        MPI_Status status[sendRequests.size()];
        MPI_Waitall(int(sendRequests.size()), sendRequests.data(), status);
    }

    checkGpuErrors(cudaFree(d_range));
    reallocate(sendScratchBuffer, oldSendSize, 1.0);
    reallocate(receiveScratchBuffer, oldRecvSize, 1.0);

    // MUST call MPI_Barrier, a collective MPI function or increment epoch before calling this function again.
}

} // namespace cstone
