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

#ifdef __CUDACC__

//! @brief time a generic unary function
template<class F>
float timeGpu(F&& f)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, cudaStreamDefault);

    f();

    cudaEventRecord(stop, cudaStreamDefault);
    cudaEventSynchronize(stop);

    float t0;
    cudaEventElapsedTime(&t0, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return t0;
}

#elif defined(__HIPCC__)

//! @brief time a generic unary function
template<class F>
float timeGpu(F&& f)
{
    hipEvent_t start, stop;
    checkGpuErrors(hipEventCreate(&start));
    checkGpuErrors(hipEventCreate(&stop));

    checkGpuErrors(hipEventRecord(start, hipStreamDefault));

    f();

    checkGpuErrors(hipEventRecord(stop, hipStreamDefault));
    checkGpuErrors(hipEventSynchronize(stop));

    float t0;
    checkGpuErrors(hipEventElapsedTime(&t0, start, stop));

    checkGpuErrors(hipEventDestroy(start));
    checkGpuErrors(hipEventDestroy(stop));

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
