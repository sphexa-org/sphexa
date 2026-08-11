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
#include <cstdlib>
#include <string>

#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "cstone/execution.hpp"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/sfc/sfc_gpu.h"

#include "timing.cuh"

using namespace cstone;

template<class KeyType>
__global__ void keysFromIntKernel(
    KeyType* keys, const uint32_t* x, const uint32_t* y, const uint32_t* z, size_t n, const AxesBits abits)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) { keys[tid] = iSfcKey<KeyType>(x[tid], y[tid], z[tid], abits); }
}

template<class KeyType>
void keysFromInt(
    cudaStream_t s, KeyType* keys, const uint32_t* x, const uint32_t* y, const uint32_t* z, size_t n, AxesBits abits)
{
    constexpr int numThreads = 256;
    keysFromIntKernel<<<iceil(n, numThreads), numThreads, 0, s>>>(keys, x, y, z, n, abits);
}

template<class KeyType>
__global__ void
decodeSfcKeysKernel(const KeyType* keys, uint32_t* x, uint32_t* y, uint32_t* z, size_t numKeys, const AxesBits axesBits)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numKeys) { util::tie(x[tid], y[tid], z[tid]) = decodeSfc(keys[tid], axesBits); }
}

template<class KeyType>
void decodeSfcKeys(
    cudaStream_t stream, const KeyType* keys, uint32_t* x, uint32_t* y, uint32_t* z, size_t numKeys, AxesBits axesBits)
{
    constexpr int numThreads = 256;
    decodeSfcKeysKernel<<<iceil(numKeys, numThreads), numThreads, 0, stream>>>(keys, x, y, z, numKeys, axesBits);
}

int main(int argc, char** argv)
{
    using IntegerType = uint64_t;
    unsigned numKeys = argc > 1 ? static_cast<unsigned>(std::stoul(argv[1])) : 32000000;

    using Real = double;
    Box<Real> box(-1, 1);
    const auto axesBits = box.getBoxDimBits(maxTreeLevel<IntegerType>{});

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

        auto computeHilbert = [&](cudaStream_t stream)
        { keysFromInt(stream, rawPtr(hilbertKeys), rawPtr(dx), rawPtr(dy), rawPtr(dz), numKeys, axesBits); };

        auto computeMorton = [&](cudaStream_t stream)
        { keysFromInt(stream, rawPtr(mortonKeys), rawPtr(dx), rawPtr(dy), rawPtr(dz), numKeys, axesBits); };

        float t_hilbert = timeGpu(computeHilbert);
        float t_morton  = timeGpu(computeMorton);
        std::cout << "compute time for " << numKeys << " hilbert keys: " << t_hilbert / 1000 << " s" << std::endl;
        std::cout << "compute time for " << numKeys << " morton keys: " << t_morton / 1000 << " s" << std::endl;

        thrust::device_vector<unsigned> dx2(numKeys);
        thrust::device_vector<unsigned> dy2(numKeys);
        thrust::device_vector<unsigned> dz2(numKeys);

        auto decodeHilbert = [&](cudaStream_t stream)
        { decodeSfcKeys(stream, rawPtr(hilbertKeys), rawPtr(dx2), rawPtr(dy2), rawPtr(dz2), numKeys, axesBits); };

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

        auto computeHilbert = [&](cudaStream_t stream)
        {
            computeSfcKeys(execution::gpuStream(stream), rawPtr(dx), rawPtr(dy), rawPtr(dz), rawPtr(hilbertKeys2),
                           numKeys, box);
        };

        auto computeMorton = [&](cudaStream_t stream)
        {
            computeSfcKeys(execution::gpuStream(stream), rawPtr(dx), rawPtr(dy), rawPtr(dz), rawPtr(mortonKeys2),
                           numKeys, box);
        };

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

        auto radixSort = [&](cudaStream_t stream)
        {
            thrust::sort_by_key(thrustExecPolicy(execution::gpuStream(stream)), (IntegerType*)rawPtr(hilbertKeys),
                                (IntegerType*)rawPtr(hilbertKeys) + numKeys, ordering.begin());
        };
        float t_radixSort = timeGpu(radixSort);

        size_t numBytesMoved = 2 * numKeys * (sizeof(IntegerType) + sizeof(unsigned));
        std::cout << "radix sort time for " << numKeys << " key-value pairs: " << t_radixSort / 1000 << " s"
                  << ", bandwidth: " << float(numBytesMoved) / t_radixSort / 1000 << " MiB/s" << std::endl;
    }
}
