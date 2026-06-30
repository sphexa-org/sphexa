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
#include "cuda_stubs.h"
#include "device_vector.h"
#include "errorcheck.cuh"

namespace cstone
{

//! @brief detection of thrust device vectors
template<class T>
struct IsDeviceVector<cstone::DeviceVector<T>> : public std::true_type
{
};

template<class T>
void memcpyH2DAsync(execution::Gpu exec, const T* src, std::size_t n, T* dest)
{
    checkGpuErrors(cudaMemcpyAsync(dest, src, sizeof(T) * n, cudaMemcpyHostToDevice, exec));
}

template<class T>
void memcpyD2HAsync(execution::Gpu exec, const T* src, std::size_t n, T* dest)
{
    checkGpuErrors(cudaMemcpyAsync(dest, src, sizeof(T) * n, cudaMemcpyDeviceToHost, exec));
}

template<class T>
void memcpyD2DAsync(execution::Gpu exec, const T* src, std::size_t n, T* dest)
{
    checkGpuErrors(cudaMemcpyAsync(dest, src, sizeof(T) * n, cudaMemcpyDeviceToDevice, exec));
}

inline void syncGpu(execution::Gpu exec) { checkGpuErrors(cudaStreamSynchronize(exec)); }

//! @brief Download DeviceVector to a host vector. Convenience function for use in testing.
template<class T>
std::vector<T> toHost(const cstone::DeviceVector<T>& v)
{
    std::vector<T> ret(v.size());
    checkGpuErrors(cudaMemcpy(ret.data(), v.data(), sizeof(T) * v.size(), cudaMemcpyDeviceToHost));
    return ret;
}

} // namespace cstone
