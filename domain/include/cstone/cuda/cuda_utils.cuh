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
void memcpyH2D(const T* src, std::size_t n, T* dest)
{
    checkGpuErrors(cudaMemcpy(dest, src, sizeof(T) * n, cudaMemcpyHostToDevice));
}

template<class T>
void memcpyD2H(const T* src, std::size_t n, T* dest)
{
    checkGpuErrors(cudaMemcpy(dest, src, sizeof(T) * n, cudaMemcpyDeviceToHost));
}

template<class T>
void memcpyD2D(const T* src, std::size_t n, T* dest)
{
    checkGpuErrors(cudaMemcpy(dest, src, sizeof(T) * n, cudaMemcpyDeviceToDevice));
}

inline void syncGpu() { checkGpuErrors(cudaDeviceSynchronize()); }

//! @brief Download DeviceVector to a host vector. Convenience function for use in testing.
template<class T>
std::vector<T> toHost(const cstone::DeviceVector<T>& v)
{
    std::vector<T> ret(v.size());
    memcpyD2H(v.data(), v.size(), ret.data());
    return ret;
}
