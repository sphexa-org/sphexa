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
