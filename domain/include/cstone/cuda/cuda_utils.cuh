/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief CUDA runtime API wrapper for compatiblity with CPU code
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <type_traits>
#include <vector>
#include "cuda_runtime.hpp"

#include "device_vector.h"
#include "cuda_stubs.h"
#include "errorcheck.cuh"

//! @brief detection of thrust device vectors
template<class T>
struct IsDeviceVector<cstone::DeviceVector<T>> : public std::true_type
{
};

template<class T>
void memcpyH2DAsync(const T* src, std::size_t n, T* dest, cudaStream_t stream)
{
    checkGpuErrors(cudaMemcpyAsync(dest, src, sizeof(T) * n, cudaMemcpyHostToDevice, stream));
}

template<class T>
void memcpyD2HAsync(const T* src, std::size_t n, T* dest, cudaStream_t stream)
{
    checkGpuErrors(cudaMemcpyAsync(dest, src, sizeof(T) * n, cudaMemcpyDeviceToHost, stream));
}

template<class T>
void memcpyD2DAsync(const T* src, std::size_t n, T* dest, cudaStream_t stream)
{
    checkGpuErrors(cudaMemcpyAsync(dest, src, sizeof(T) * n, cudaMemcpyDeviceToDevice, stream));
}

//! @brief Wait for all work on @p stream to complete
inline void syncGpu(cudaStream_t stream) { checkGpuErrors(cudaStreamSynchronize(stream)); }

//! @brief Download DeviceVector to a host vector. Convenience function for use in testing.
template<class T>
std::vector<T> toHost(const cstone::DeviceVector<T>& v)
{
    std::vector<T> ret(v.size());
    memcpyD2HAsync(v.data(), v.size(), ret.data(), 0);
    syncGpu(0);
    return ret;
}
