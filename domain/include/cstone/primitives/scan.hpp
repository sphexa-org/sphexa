/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Parallel prefix sum
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cstone/primitives/stl.hpp"

namespace cstone
{

/*! @brief multi-threaded exclusive scan (prefix sum) implementation
 *
 * @tparam T1, T2       integer types
 * @param in            input values, length = @p numElements
 * @param out           output values, length = @p numElements
 * @param numElements
 */
template<class T1, class T2>
void exclusiveScan(const T1* in, T2* out, size_t numElements)
{
    constexpr int blockSize = (8192 + 16384) / sizeof(T1);

    int numThreads = 1;
#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp single
        numThreads = omp_get_num_threads();
    }
#endif

    T2 superBlock[2][numThreads + 1];
    std::fill(superBlock[0], superBlock[0] + numThreads + 1, 0);
    std::fill(superBlock[1], superBlock[1] + numThreads + 1, 0);

    unsigned elementsPerStep = numThreads * blockSize;
    unsigned nSteps          = numElements / elementsPerStep;

#pragma omp parallel num_threads(numThreads)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        for (size_t step = 0; step < nSteps; ++step)
        {
            size_t stepOffset = step * elementsPerStep + tid * blockSize;

            std::exclusive_scan(in + stepOffset, in + stepOffset + blockSize, out + stepOffset, 0);

            superBlock[step % 2][tid] = out[stepOffset + blockSize - 1] + in[stepOffset + blockSize - 1];

#pragma omp barrier

            T2 tSum = superBlock[(step + 1) % 2][numThreads];
            for (int t = 0; t < tid; ++t)
                tSum += superBlock[step % 2][t];

            if (tid == numThreads - 1) superBlock[step % 2][numThreads] = tSum + superBlock[step % 2][numThreads - 1];

            std::for_each(out + stepOffset, out + stepOffset + blockSize, [shift = tSum](T2& val) { val += shift; });
        }
    }

    // remainder
    T2 stepSum = superBlock[(nSteps + 1) % 2][numThreads];
    std::exclusive_scan(in + nSteps * elementsPerStep, in + numElements, out + nSteps * elementsPerStep, stepSum);
}

template<class T>
T exclusiveScanSerialInplace(T* out, size_t num_elements, T init)
{
    T a = init;
    T b = init;
    for (size_t i = 0; i < num_elements; ++i)
    {
        a += out[i];
        out[i] = b;
        b      = a;
    }
    return b;
}

#ifdef _OPENMP

template<class T>
void exclusiveScan(T* out, size_t numElements)
{
    constexpr int blockSize = (2 * 16384) / sizeof(T);

    int numThreads = 1;
#pragma omp parallel
    {
#pragma omp single
        numThreads = omp_get_num_threads();
    }

    T superBlock[2][numThreads + 1];
    std::fill(superBlock[0], superBlock[0] + numThreads + 1, 0);
    std::fill(superBlock[1], superBlock[1] + numThreads + 1, 0);

    unsigned elementsPerStep = numThreads * blockSize;
    unsigned nSteps          = numElements / elementsPerStep;

#pragma omp parallel num_threads(numThreads)
    {
        int tid = omp_get_thread_num();
        for (size_t step = 0; step < nSteps; ++step)
        {
            size_t stepOffset = step * elementsPerStep + tid * blockSize;

            superBlock[step % 2][tid] = exclusiveScanSerialInplace(out + stepOffset, blockSize, T(0));

#pragma omp barrier

            T tSum = superBlock[(step + 1) % 2][numThreads];
            for (int t = 0; t < tid; ++t)
                tSum += superBlock[step % 2][t];

            if (tid == numThreads - 1) superBlock[step % 2][numThreads] = tSum + superBlock[step % 2][numThreads - 1];

            std::for_each(out + stepOffset, out + stepOffset + blockSize, [shift = tSum](T& val) { val += shift; });
        }
    }

    // remainder
    T stepSum = superBlock[(nSteps + 1) % 2][numThreads];
    exclusiveScanSerialInplace(out + nSteps * elementsPerStep, numElements - nSteps * elementsPerStep, stepSum);
}

#else

template<class T>
void exclusiveScan(T* out, size_t numElements)
{
    exclusiveScanSerialInplace(out, numElements, T(0));
}

#endif

} // namespace cstone
