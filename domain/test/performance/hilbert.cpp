/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Benchmark cornerstone octree generation on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

#define THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_OMP
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "cstone/primitives/gather.hpp"
#include "cstone/sfc/sfc.hpp"

using namespace cstone;

int main()
{
    using KeyType    = uint64_t;
    unsigned numKeys = 32000000;

    using Real = double;
    Box<Real> box(-1, 1);

    std::mt19937 gen;
    std::uniform_real_distribution<Real> distribution(box.xmin(), box.xmax());
    auto getRand = [&distribution, &gen]() { return distribution(gen); };

    std::vector<Real> x(numKeys);
    std::vector<Real> y(numKeys);
    std::vector<Real> z(numKeys);

    std::generate(begin(x), end(x), getRand);
    std::generate(begin(y), end(y), getRand);
    std::generate(begin(z), end(z), getRand);

    std::vector<KeyType> sfcKeys(numKeys);

    {
        auto cpu_t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < numKeys; ++i)
        {
            sfcKeys[i] = sfc3D<MortonKey<KeyType>>(x[i], y[i], z[i], box);
        }
        auto cpu_t1            = std::chrono::high_resolution_clock::now();
        double cpu_time_morton = std::chrono::duration<double>(cpu_t1 - cpu_t0).count();

        std::cout << "compute time for " << numKeys << " morton keys: " << cpu_time_morton << " s on CPU" << std::endl;
    }

    {
        auto cpu_t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < numKeys; ++i)
        {
            sfcKeys[i] = sfc3D<HilbertKey<KeyType>>(x[i], y[i], z[i], box);
        }
        auto cpu_t1             = std::chrono::high_resolution_clock::now();
        double cpu_time_hilbert = std::chrono::duration<double>(cpu_t1 - cpu_t0).count();

        std::cout << "compute time for " << numKeys << " hilbert keys: " << cpu_time_hilbert << " s on CPU"
                  << std::endl;
    }

    {
        std::vector<unsigned> ordering(numKeys);
        std::iota(ordering.begin(), ordering.end(), 0);

        auto cpu_t0 = std::chrono::high_resolution_clock::now();
        thrust::sort_by_key(thrust::host, sfcKeys.begin(), sfcKeys.end(), ordering.begin());
        auto cpu_t1          = std::chrono::high_resolution_clock::now();
        double cpu_time_sort = std::chrono::duration<double>(cpu_t1 - cpu_t0).count();

        size_t numBytesMoved = 2 * numKeys * (sizeof(KeyType) + sizeof(unsigned));
        std::cout << "radix sort time for " << numKeys << ": " << cpu_time_sort << ", bandwidth "
                  << double(numBytesMoved) / 1e6 / cpu_time_sort << " MiB/s" << std::endl;
    }
}
