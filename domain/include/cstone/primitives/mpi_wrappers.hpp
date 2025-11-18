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

#include <mpi.h>
#include <algorithm>
#include <cassert>
#include <limits>
#include <type_traits>
#include <vector>

template<class T>
struct MpiType
{
};

template<>
struct MpiType<double>
{
    operator MPI_Datatype() const noexcept { return MPI_DOUBLE; }
};

template<>
struct MpiType<float>
{
    operator MPI_Datatype() const noexcept { return MPI_FLOAT; }
};

template<>
struct MpiType<char>
{
    operator MPI_Datatype() const noexcept { return MPI_CHAR; }
};

template<>
struct MpiType<unsigned char>
{
    operator MPI_Datatype() const noexcept { return MPI_UNSIGNED_CHAR; }
};

template<>
struct MpiType<short>
{
    operator MPI_Datatype() const noexcept { return MPI_SHORT; }
};

template<>
struct MpiType<unsigned short>
{
    operator MPI_Datatype() const noexcept { return MPI_UNSIGNED_SHORT; }
};

template<>
struct MpiType<int>
{
    operator MPI_Datatype() const noexcept { return MPI_INT; }
};

template<>
struct MpiType<unsigned>
{
    operator MPI_Datatype() const noexcept { return MPI_UNSIGNED; }
};

template<>
struct MpiType<long>
{
    operator MPI_Datatype() const noexcept { return MPI_LONG; }
};

template<>
struct MpiType<unsigned long>
{
    operator MPI_Datatype() const noexcept { return MPI_UNSIGNED_LONG; }
};

template<>
struct MpiType<unsigned long long>
{
    operator MPI_Datatype() const noexcept { return MPI_UNSIGNED_LONG_LONG; }
};

template<class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
auto mpiSendAsync(T* data, size_t count, int rank, int tag, std::vector<MPI_Request>& requests)
{
    assert(count <= std::numeric_limits<int>::max());
    requests.push_back(MPI_Request{});
    return MPI_Isend(data, int(count), MpiType<std::decay_t<T>>{}, rank, tag, MPI_COMM_WORLD, &requests.back());
}

//! @brief adaptor to wrap compile-time size arrays into flattened arrays of the underlying type
template<class T, std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
auto mpiSendAsync(T* data, size_t count, int rank, int tag, std::vector<MPI_Request>& requests)
{
    using ValueType    = typename T::value_type;
    constexpr size_t N = T{}.size();
    ValueType* ptr     = reinterpret_cast<ValueType*>(data);

    return mpiSendAsync(ptr, count * N, rank, tag, requests);
}

//! @brief Send char buffers as type T to mitigate the 32-bit send count limitation of MPI
template<class T>
auto mpiSendAsyncAs(char* data, size_t numBytes, int rank, int tag, std::vector<MPI_Request>& requests)
{
    return mpiSendAsync(reinterpret_cast<T*>(data), numBytes / sizeof(T), rank, tag, requests);
}

template<class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
auto mpiRecvSync(T* data, int count, int rank, int tag, MPI_Status* status)
{
    return MPI_Recv(data, count, MpiType<std::decay_t<T>>{}, rank, tag, MPI_COMM_WORLD, status);
}

//! @brief adaptor to wrap compile-time size arrays into flattened arrays of the underlying type
template<class T, std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
auto mpiRecvSync(T* data, int count, int rank, int tag, MPI_Status* status)
{
    using ValueType    = typename T::value_type;
    constexpr size_t N = T{}.size();
    ValueType* ptr     = reinterpret_cast<ValueType*>(data);

    return mpiRecvSync(ptr, count * N, rank, tag, status);
}

template<class T>
auto mpiRecvSyncAs(char* data, size_t numBytes, int rank, int tag, MPI_Status* status)
{
    return mpiRecvSync(reinterpret_cast<T*>(data), numBytes / sizeof(T), rank, tag, status);
}

template<class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
auto mpiRecvAsync(T* data, int count, int rank, int tag, std::vector<MPI_Request>& requests)
{
    requests.push_back(MPI_Request{});
    return MPI_Irecv(data, count, MpiType<std::decay_t<T>>{}, rank, tag, MPI_COMM_WORLD, &requests.back());
}

//! @brief adaptor to wrap compile-time size arrays into flattened arrays of the underlying type
template<class T, std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
auto mpiRecvAsync(T* data, int count, int rank, int tag, std::vector<MPI_Request>& requests)
{
    using ValueType    = typename T::value_type;
    constexpr size_t N = T{}.size();
    ValueType* ptr     = reinterpret_cast<ValueType*>(data);

    return mpiRecvAsync(ptr, count * N, rank, tag, requests);
}

//! @brief MPI_Get_count for standard types
template<class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
auto mpiGetCount(MPI_Status* status, int* count)
{
    return MPI_Get_count(status, MpiType<T>{}, count);
}

//! @brief MPI_Get_count for arrays of standard types, T = std::array<ValueType, N>
template<class T, std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
auto mpiGetCount(MPI_Status* status, int* count)
{
    using ValueType    = typename T::value_type;
    constexpr size_t N = T{}.size();
    static_assert(N && "array length must be nonzero");
    auto ret = MPI_Get_count(status, MpiType<ValueType>{}, count);
    *count /= N;
    return ret;
}

template<class Ts, class Td, std::enable_if_t<std::is_arithmetic_v<Td>, int> = 0>
auto mpiAllreduce(const Ts* src, Td* dest, int count, MPI_Op op, MPI_Comm comm)
{
    return MPI_Allreduce(src, dest, count, MpiType<Td>{}, op, comm);
}

//! @brief adaptor to wrap compile-time size arrays into flattened arrays of the underlying type
template<class Ts, class Td, std::enable_if_t<!std::is_arithmetic_v<Td>, int> = 0>
auto mpiAllreduce(const Ts* src, Td* dest, int count, MPI_Op op, MPI_Comm comm)
{
    using ValueType    = typename Td::value_type;
    constexpr size_t N = Td{}.size();

    using SrcType = std::conditional_t<std::is_same_v<void, Ts>, void, ValueType>;
    auto src_ptr  = reinterpret_cast<const SrcType*>(src);
    auto dest_ptr = reinterpret_cast<ValueType*>(dest);

    return mpiAllreduce(src_ptr, dest_ptr, count * N, op, comm);
}

template<class Ts, class Td, std::enable_if_t<std::is_arithmetic_v<Td>, int> = 0>
auto mpiAllgatherv(const Ts* src, int sendCount, Td* dest, const int* counts, const int* displ, MPI_Comm comm)
{
    return MPI_Allgatherv(src, sendCount, MpiType<Td>{}, dest, counts, displ, MpiType<Td>{}, comm);
}

//! @brief adaptor to wrap compile-time size arrays into flattened arrays of the underlying type
template<class Ts, class Td, std::enable_if_t<!std::is_arithmetic_v<Td>, int> = 0>
auto mpiAllgatherv(const Ts* src, int sendCount, Td* dest, const int* counts, const int* displ, MPI_Comm comm)
{
    using ValueType    = typename Td::value_type;
    constexpr size_t N = Td{}.size();

    using SrcType = std::conditional_t<std::is_same_v<void, Ts>, void, ValueType>;
    auto src_ptr  = reinterpret_cast<const SrcType*>(src);
    auto dest_ptr = reinterpret_cast<ValueType*>(dest);

    int numRanks;
    MPI_Comm_size(comm, &numRanks);

    std::vector<int> countsN(numRanks), displN(numRanks);
    std::transform(counts, counts + numRanks, countsN.data(), [N](auto x) { return x * N; });
    std::transform(displ, displ + numRanks, displN.data(), [N](auto x) { return x * N; });
    return mpiAllgatherv(src_ptr, sendCount * N, dest_ptr, countsN.data(), displN.data(), comm);
}
