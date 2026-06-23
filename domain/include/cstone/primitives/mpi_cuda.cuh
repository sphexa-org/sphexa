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

#include "cstone/execution.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/util/noinit_alloc.hpp"
#include "cstone/cuda/cuda_stubs.h"

namespace cstone
{

#ifdef CSTONE_HAVE_GPU_AWARE_MPI
constexpr inline bool useGpuDirect = true;
#else
constexpr inline bool useGpuDirect = false;
#endif

template<class T>
auto mpiSendGpuDirect(execution::Gpu exec,
                      T* data,
                      size_t count,
                      int rank,
                      int tag,
                      std::vector<MPI_Request>& requests,
                      [[maybe_unused]] std::vector<std::vector<T, util::DefaultInitAdaptor<T>>>& buffers,
                      MPI_Comm comm)
{
    if constexpr (!useGpuDirect)
    {
        std::vector<T, util::DefaultInitAdaptor<T>> hostBuffer(count);
        memcpyD2HAsync(exec, data, count, hostBuffer.data());
        syncGpu(exec);
        auto errCode = mpiSendAsync(hostBuffer.data(), count, rank, tag, requests, comm);
        buffers.push_back(std::move(hostBuffer));

        return errCode;
    }
    else
    {
        syncGpu(exec);
        return mpiSendAsync(data, count, rank, tag, requests, comm);
    }
}

//! @brief Send char buffers cast to a transfer type @p T to mitigate the 32-bit send count limitation of MPI
template<class T, std::enable_if_t<!std::is_same_v<T, char>, int> = 0>
auto mpiSendGpuDirect(execution::Gpu exec,
                      char* data,
                      size_t numBytes,
                      int rank,
                      int tag,
                      std::vector<MPI_Request>& requests,
                      std::vector<std::vector<T, util::DefaultInitAdaptor<T>>>& buffers,
                      MPI_Comm comm)
{
    return mpiSendGpuDirect(exec, reinterpret_cast<T*>(data), numBytes / sizeof(T), rank, tag, requests, buffers, comm);
}

template<class T>
auto mpiRecvGpuDirect(execution::Gpu exec, T* data, int count, int rank, int tag, MPI_Status* status, MPI_Comm comm)
{
    if constexpr (!useGpuDirect)
    {
        std::vector<T, util::DefaultInitAdaptor<T>> hostBuffer(count);
        auto errCode = mpiRecvSync(hostBuffer.data(), count, rank, tag, status, comm);
        memcpyH2DAsync(exec, hostBuffer.data(), count, data);
        syncGpu(exec);

        return errCode;
    }
    else
    {
        syncGpu(exec);
        return mpiRecvSync(data, count, rank, tag, status, comm);
    }
}

//! @brief this wrapper is needed to support sending from GPU buffers with staging through host (no GPU-direct MPI)
template<class T>
auto mpiSendAsyncAcc(execution::Cpu,
                     T* data,
                     size_t count,
                     int rank,
                     int tag,
                     std::vector<MPI_Request>& requests,
                     [[maybe_unused]] std::vector<std::vector<T, util::DefaultInitAdaptor<T>>>& buffers,
                     MPI_Comm comm)
{
    mpiSendAsync(data, count, rank, tag, requests, comm);
}

template<class T>
auto mpiSendAsyncAcc(execution::Gpu exec,
                     T* data,
                     size_t count,
                     int rank,
                     int tag,
                     std::vector<MPI_Request>& requests,
                     [[maybe_unused]] std::vector<std::vector<T, util::DefaultInitAdaptor<T>>>& buffers,
                     MPI_Comm comm)
{
    mpiSendGpuDirect(exec, data, count, rank, tag, requests, buffers, comm);
}

//! @brief this wrapper is needed to support receiving into GPU buffers with staging through host (no GPU-direct MPI)
template<class T>
auto mpiRecvSyncAcc(execution::Cpu, T* data, int count, int rank, int tag, MPI_Status* status, MPI_Comm comm)
{
    mpiRecvSync(data, count, rank, tag, status, comm);
}

template<class T>
auto mpiRecvSyncAcc(execution::Gpu exec, T* data, int count, int rank, int tag, MPI_Status* status, MPI_Comm comm)
{
    mpiRecvGpuDirect(exec, data, count, rank, tag, status, comm);
}

template<class T>
auto mpiAllreduceGpuDirect(execution::Gpu exec, const T* src, T* dest, size_t count, MPI_Op op, MPI_Comm comm)
{
    if constexpr (!useGpuDirect)
    {
        std::vector<T> srcBuf(count), destBuf(count);
        memcpyD2HAsync(exec, src, count, srcBuf.data());
        syncGpu(exec);
        mpiAllreduce(srcBuf.data(), destBuf.data(), count, op, comm);
        memcpyH2DAsync(exec, destBuf.data(), count, dest);
        syncGpu(exec);
    }
    else
    {
        syncGpu(exec);
        mpiAllreduce(src, dest, count, op, comm);
    }
}

//! @brief adaptor to wrap compile-time size arrays into flattened arrays of the underlying type
template<class Ts, class Td>
auto mpiAllgathervGpuDirect(
    execution::Cpu, const Ts* src, int sendCount, Td* dest, const int* counts, const int* displ, MPI_Comm comm)
{
    mpiAllgatherv(src, sendCount, dest, counts, displ, comm);
}

template<class Ts, class Td>
auto mpiAllgathervGpuDirect(
    execution::Gpu exec, const Ts* src, int sendCount, Td* dest, const int* counts, const int* displ, MPI_Comm comm)
{
    if constexpr (!useGpuDirect)
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
                memcpyD2HAsync(exec, src, numElements, srcUse);
                syncGpu(exec);
            }
        }

        std::vector<Td> destStage(numElements);
        if (src == MPI_IN_PLACE)
        {
            memcpyD2HAsync(exec, dest, numElements, destStage.data());
            syncGpu(exec);
        }
        mpiAllgatherv(srcUse, sendCount, destStage.data(), counts, displ, comm);
        memcpyH2DAsync(exec, destStage.data(), destStage.size(), dest);
        syncGpu(exec);
    }
    else
    {
        syncGpu(exec);
        mpiAllgatherv(src, sendCount, dest, counts, displ, comm);
    }
}

} // namespace cstone
