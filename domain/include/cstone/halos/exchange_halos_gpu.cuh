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
                     const SendList& incomingHalos,
                     const SendList& outgoingHalos,
                     DevVec1& sendScratchBuffer,
                     DevVec2& receiveScratchBuffer,
                     Arrays... arrays)
{
    constexpr int alignment     = 8;
    using IndexType             = SendManifest::IndexType;
    const float allocGrowthRate = 1.05;

    int haloExchangeTag = static_cast<int>(P2pTags::haloExchange) + epoch;
    std::vector<MPI_Request> sendRequests;
    std::vector<std::vector<char, util::DefaultInitAdaptor<char>>> sendBuffers;

    const size_t oldSendSize = reallocateBytes(
        sendScratchBuffer, computeTotalSendBytes<alignment>(outgoingHalos, -1, 0, arrays...), allocGrowthRate);

    size_t numRanges = std::max(maxNumRanges(outgoingHalos), maxNumRanges(incomingHalos));
    IndexType* d_range;
    checkGpuErrors(cudaMalloc((void**)&d_range, 2 * numRanges * sizeof(IndexType)));
    IndexType* d_rangeScan = d_range + numRanges;

    char* sendPtr = reinterpret_cast<char*>(rawPtr(sendScratchBuffer));
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

        size_t numBytesSend = util::computeByteOffsets(sendCount, alignment, arrays...).back();
        mpiSendGpuDirect(sendPtr, numBytesSend, int(destinationRank), haloExchangeTag, sendRequests, sendBuffers);
        sendPtr += numBytesSend;
    }

    int numMessages       = 0;
    size_t maxReceiveSize = 0;
    for (const auto& incomingHalo : incomingHalos)
    {
        numMessages += int(incomingHalo.totalCount() > 0);
        maxReceiveSize = std::max(maxReceiveSize, incomingHalo.totalCount());
    }
    size_t maxReceiveBytes = util::computeByteOffsets(maxReceiveSize, alignment, arrays...).back();

    const size_t oldRecvSize = reallocateBytes(receiveScratchBuffer, maxReceiveBytes, allocGrowthRate);
    char* receiveBuffer      = reinterpret_cast<char*>(rawPtr(receiveScratchBuffer));

    while (numMessages > 0)
    {
        MPI_Status status;
        mpiRecvGpuDirect(receiveBuffer, maxReceiveBytes, MPI_ANY_SOURCE, haloExchangeTag, &status);
        int receiveRank     = status.MPI_SOURCE;
        const auto& inHalos = incomingHalos[receiveRank];
        size_t receiveCount = inHalos.totalCount();

        // compute indices to extract and upload to GPU
        checkGpuErrors(
            cudaMemcpy(d_range, inHalos.offsets(), inHalos.nRanges() * sizeof(IndexType), cudaMemcpyHostToDevice));
        checkGpuErrors(
            cudaMemcpy(d_rangeScan, inHalos.scan(), inHalos.nRanges() * sizeof(IndexType), cudaMemcpyHostToDevice));

        auto scatterArray = [d_range, d_rangeScan, numRanges = inHalos.nRanges(), receiveCount](auto arrayPtr)
        { scatterRanges(d_rangeScan, d_range, numRanges, arrayPtr[0], arrayPtr[1], receiveCount); };

        for_each_tuple(scatterArray, util::packBufferPtrs<alignment>(receiveBuffer, receiveCount, arrays...));
        checkGpuErrors(cudaDeviceSynchronize());

        numMessages--;
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
