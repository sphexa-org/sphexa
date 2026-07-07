/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief GPU timing utility
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <chrono>
#include "cstone/cuda/errorcheck.cuh"

#if defined(__CUDACC__) || defined(__HIP__)

#include "cstone/cuda/cuda_runtime.hpp"

//! @brief time a generic unary function
template<class F>
float timeGpu(F&& f)
{
    cudaEvent_t start, stop;
    checkGpuErrors(cudaEventCreate(&start));
    checkGpuErrors(cudaEventCreate(&stop));
    cudaStream_t stream;
    checkGpuErrors(cudaStreamCreate(&stream));

    checkGpuErrors(cudaEventRecord(start, stream));

    f(stream);

    checkGpuErrors(cudaEventRecord(stop, stream));
    checkGpuErrors(cudaEventSynchronize(stop));

    float t0;
    checkGpuErrors(cudaEventElapsedTime(&t0, start, stop));

    checkGpuErrors(cudaStreamDestroy(stream));
    checkGpuErrors(cudaEventDestroy(start));
    checkGpuErrors(cudaEventDestroy(stop));

    return t0;
}

#endif

template<class F>
float timeCpu(F&& f)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    f();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float>(t1 - t0).count();
}
