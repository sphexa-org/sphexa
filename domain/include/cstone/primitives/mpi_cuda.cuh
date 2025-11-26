/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  A few C++ wrappers for MPI C functions
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <vector>

#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/util/noinit_alloc.hpp"
#include "cstone/cuda/cuda_stubs.h"

#ifdef CSTONE_HAVE_GPU_AWARE_MPI
constexpr inline bool useGpuDirect = true;
#else
constexpr inline bool useGpuDirect = false;
#endif

template<class T>
auto mpiSendGpuDirect(T* data,
                      size_t count,
                      int rank,
                      int tag,
                      std::vector<MPI_Request>& requests,
                      [[maybe_unused]] std::vector<std::vector<T, util::DefaultInitAdaptor<T>>>& buffers)
{
    if constexpr (!useGpuDirect)
    {
        std::vector<T, util::DefaultInitAdaptor<T>> hostBuffer(count);
        memcpyD2H(data, count, hostBuffer.data());
        auto errCode = mpiSendAsync(hostBuffer.data(), count, rank, tag, requests);
        buffers.push_back(std::move(hostBuffer));

        return errCode;
    }
    else { return mpiSendAsync(data, count, rank, tag, requests); }
}

//! @brief Send char buffers cast to a transfer type @p T to mitigate the 32-bit send count limitation of MPI
template<class T, std::enable_if_t<!std::is_same_v<T, char>, int> = 0>
auto mpiSendGpuDirect(char* data,
                      size_t numBytes,
                      int rank,
                      int tag,
                      std::vector<MPI_Request>& requests,
                      std::vector<std::vector<T, util::DefaultInitAdaptor<T>>>& buffers)
{
    return mpiSendGpuDirect(reinterpret_cast<T*>(data), numBytes / sizeof(T), rank, tag, requests, buffers);
}

template<class T>
auto mpiRecvGpuDirect(T* data, int count, int rank, int tag, MPI_Status* status)
{
    if constexpr (!useGpuDirect)
    {
        std::vector<T, util::DefaultInitAdaptor<T>> hostBuffer(count);
        auto errCode = mpiRecvSync(hostBuffer.data(), count, rank, tag, status);
        memcpyH2D(hostBuffer.data(), count, data);

        return errCode;
    }
    else { return mpiRecvSync(data, count, rank, tag, status); }
}

//! @brief this wrapper is needed to support sending from GPU buffers with staging through host (no GPU-direct MPI)
template<bool useGpu, class T>
auto mpiSendAsyncAcc(T* data,
                     size_t count,
                     int rank,
                     int tag,
                     std::vector<MPI_Request>& requests,
                     [[maybe_unused]] std::vector<std::vector<T, util::DefaultInitAdaptor<T>>>& buffers)
{
    if constexpr (useGpu) { mpiSendGpuDirect(data, count, rank, tag, requests, buffers); }
    else { mpiSendAsync(data, count, rank, tag, requests); }
}

//! @brief this wrapper is needed to support sending from GPU buffers with staging through host (no GPU-direct MPI)
template<bool useGpu, class T>
auto mpiRecvSyncAcc(T* data, int count, int rank, int tag, MPI_Status* status)
{
    if constexpr (useGpu) { mpiRecvGpuDirect(data, count, rank, tag, status); }
    else { mpiRecvSync(data, count, rank, tag, status); }
}

template<class T>
auto mpiAllreduceGpuDirect(const T* src, T* dest, size_t count, MPI_Op op, MPI_Comm comm)
{
    if constexpr (!useGpuDirect)
    {
        std::vector<T> srcBuf(count), destBuf(count);
        memcpyD2H(src, count, srcBuf.data());
        mpiAllreduce(srcBuf.data(), destBuf.data(), count, op, comm);
        memcpyH2D(destBuf.data(), count, dest);
    }
    else { mpiAllreduce(src, dest, count, op, comm); }
}

//! @brief adaptor to wrap compile-time size arrays into flattened arrays of the underlying type
template<bool useGpu, class Ts, class Td>
auto mpiAllgathervGpuDirect(const Ts* src, int sendCount, Td* dest, const int* counts, const int* displ, MPI_Comm comm)
{
    if constexpr (useGpu && !useGpuDirect)
    {
        int numRanks;
        MPI_Comm_size(comm, &numRanks);
        std::size_t numElements = displ[numRanks - 1] + counts[numRanks - 1];

        Ts* srcUse = reinterpret_cast<Ts*>(MPI_IN_PLACE);
        std::vector<char> srcStage;
        if constexpr (not std::is_same_v<Ts, void>)
        {
            if (src != MPI_IN_PLACE)
            {
                srcStage.resize(sizeof(Ts) * numElements);
                srcUse = reinterpret_cast<Ts*>(srcStage.data());
                memcpyD2H(src, numElements, srcUse);
            }
        }

        std::vector<Td> destStage(numElements);
        if (src == MPI_IN_PLACE) { memcpyD2H(dest, numElements, destStage.data()); }
        mpiAllgatherv(srcUse, sendCount, destStage.data(), counts, displ, comm);
        memcpyH2D(destStage.data(), destStage.size(), dest);
    }
    else
    {
        mpiAllgatherv(src, sendCount, dest, counts, displ, comm);
    }
}
