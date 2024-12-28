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

#include <iostream>
#include <random>

#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/sfc/sfc_gpu.h"

#include "timing.cuh"

using namespace cstone;

template<class KeyType>
__global__ void
computeSfcKeysKernel(KeyType* keys, const unsigned* x, const unsigned* y, const unsigned* z, size_t numKeys)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numKeys) { keys[tid] = iSfcKey<KeyType>(x[tid], y[tid], z[tid]); }
}

template<class KeyType>
inline void computeSfcKeys(KeyType* keys, const unsigned* x, const unsigned* y, const unsigned* z, size_t numKeys)
{
    constexpr int threadsPerBlock = 256;
    computeSfcKeysKernel<<<iceil(numKeys, threadsPerBlock), threadsPerBlock>>>(keys, x, y, z, numKeys);
}

template<class KeyType>
__global__ void decodeSfcKeysKernel(const KeyType* keys, unsigned* x, unsigned* y, unsigned* z, size_t numKeys)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numKeys) { thrust::tie(x[tid], y[tid], z[tid]) = decodeSfc(keys[tid]); }
}

template<class KeyType>
inline void decodeSfcKeys(const KeyType* keys, unsigned* x, unsigned* y, unsigned* z, size_t numKeys)
{
    constexpr int threadsPerBlock = 256;
    decodeSfcKeysKernel<<<iceil(numKeys, threadsPerBlock), threadsPerBlock>>>(keys, x, y, z, numKeys);
}

int main()
{
    using IntegerType = uint64_t;
    unsigned numKeys  = 32000000;

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

    thrust::device_vector<MortonKey<IntegerType>> mortonKeys(numKeys);
    thrust::device_vector<HilbertKey<IntegerType>> hilbertKeys(numKeys);

    {
        std::vector<unsigned> ix(numKeys);
        std::vector<unsigned> iy(numKeys);
        std::vector<unsigned> iz(numKeys);

        auto normIntX = [&box](Real a) { return toNBitInt<IntegerType>(normalize(a, box.xmin(), box.xmax())); };
        auto normIntY = [&box](Real a) { return toNBitInt<IntegerType>(normalize(a, box.ymin(), box.ymax())); };
        auto normIntZ = [&box](Real a) { return toNBitInt<IntegerType>(normalize(a, box.zmin(), box.zmax())); };
        std::transform(begin(x), end(x), begin(ix), normIntX);
        std::transform(begin(y), end(y), begin(iy), normIntY);
        std::transform(begin(z), end(z), begin(iz), normIntZ);

        thrust::device_vector<unsigned> dx = ix;
        thrust::device_vector<unsigned> dy = iy;
        thrust::device_vector<unsigned> dz = iz;

        auto computeHilbert = [&]()
        { computeSfcKeys(rawPtr(hilbertKeys), rawPtr(dx), rawPtr(dy), rawPtr(dz), numKeys); };

        auto computeMorton = [&]() { computeSfcKeys(rawPtr(mortonKeys), rawPtr(dx), rawPtr(dy), rawPtr(dz), numKeys); };

        float t_hilbert = timeGpu(computeHilbert);
        float t_morton  = timeGpu(computeMorton);
        std::cout << "compute time for " << numKeys << " hilbert keys: " << t_hilbert / 1000 << " s" << std::endl;
        std::cout << "compute time for " << numKeys << " morton keys: " << t_morton / 1000 << " s" << std::endl;

        thrust::device_vector<unsigned> dx2(numKeys);
        thrust::device_vector<unsigned> dy2(numKeys);
        thrust::device_vector<unsigned> dz2(numKeys);

        auto decodeHilbert = [&]()
        { decodeSfcKeys(rawPtr(hilbertKeys), rawPtr(dx2), rawPtr(dy2), rawPtr(dz2), numKeys); };

        float t_decode  = timeGpu(decodeHilbert);
        bool passDecode = thrust::equal(dx.begin(), dx.end(), dx2.begin()) &&
                          thrust::equal(dy.begin(), dy.end(), dy2.begin()) &&
                          thrust::equal(dz.begin(), dz.end(), dz2.begin());
        std::string result = (passDecode) ? "pass" : "fail";
        std::cout << "decode time for " << numKeys << " hilbert keys: " << t_decode / 1000 << " s, result: " << result
                  << std::endl;
    }

    thrust::device_vector<MortonKey<IntegerType>> mortonKeys2(numKeys);
    thrust::device_vector<HilbertKey<IntegerType>> hilbertKeys2(numKeys);

    {
        thrust::device_vector<Real> dx = x;
        thrust::device_vector<Real> dy = y;
        thrust::device_vector<Real> dz = z;

        auto computeHilbert = [&]()
        { computeSfcKeysGpu(rawPtr(dx), rawPtr(dy), rawPtr(dz), rawPtr(hilbertKeys2), numKeys, box); };

        auto computeMorton = [&]()
        { computeSfcKeysGpu(rawPtr(dx), rawPtr(dy), rawPtr(dz), rawPtr(mortonKeys2), numKeys, box); };

        float t_hilbert = timeGpu(computeHilbert);
        float t_morton  = timeGpu(computeMorton);
        std::cout << "compute time for " << numKeys << " hilbert keys from doubles : " << t_hilbert / 1000 << " s"
                  << std::endl;
        std::cout << "compute time for " << numKeys << " morton keys from doubles: " << t_morton / 1000 << " s"
                  << std::endl;
    }

    std::cout << "keys match: " << thrust::equal(hilbertKeys.begin(), hilbertKeys.end(), hilbertKeys2.begin())
              << std::endl;

    {
        thrust::device_vector<unsigned> ordering(numKeys);
        thrust::sequence(ordering.begin(), ordering.end(), 0);

        auto radixSort = [&]()
        {
            thrust::sort_by_key(thrust::device, (IntegerType*)rawPtr(hilbertKeys),
                                (IntegerType*)rawPtr(hilbertKeys) + numKeys, ordering.begin());
        };
        float t_radixSort = timeGpu(radixSort);

        size_t numBytesMoved = 2 * numKeys * (sizeof(IntegerType) + sizeof(unsigned));
        std::cout << "radix sort time for " << numKeys << " key-value pairs: " << t_radixSort / 1000 << " s"
                  << ", bandwidth: " << float(numBytesMoved) / t_radixSort / 1000 << " MiB/s" << std::endl;
    }
}
