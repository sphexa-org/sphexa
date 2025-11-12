/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  SFC encoding/decoding in 32- and 64-bit on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/cuda/errorcheck.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/sfc/sfc_gpu.h"

namespace cstone
{

template<class KeyType, class T>
__global__ void
computeSfcKeysKernel(KeyType* keys, const T* x, const T* y, const T* z, size_t numKeys, const Box<T> box)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numKeys)
    {
        KeyType rmKey = removeKey<KeyType>{};
        if (keys[tid] != rmKey) { keys[tid] = sfc3D<KeyType>(x[tid], y[tid], z[tid], box); }
    }
}

template<class KeyType, class T>
__global__ void
computeSfcMixDKeysKernel(KeyType* keys, const T* x, const T* y, const T* z, size_t numKeys, const Box<T> box, unsigned bx,
                        unsigned by, unsigned bz)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numKeys)
    {
        KeyType rmKey = removeKey<KeyType>{};
        if (keys[tid] != rmKey) { keys[tid] = sfcMixD<KeyType>(x[tid], y[tid], z[tid], box, bx, by, bz); }
    }
}

template<class KeyType, class T>
void computeSfcKeysGpu(const T* x, const T* y, const T* z, KeyType* keys, size_t numKeys, const Box<T>& box)
{
    if (numKeys == 0) { return; }

    constexpr int threadsPerBlock = 256;
    computeSfcKeysKernel<<<iceil(numKeys, threadsPerBlock), threadsPerBlock>>>(keys, x, y, z, numKeys, box);
    checkGpuErrors(cudaGetLastError());
}

template<class KeyType, class T>
void computeSfcMixDKeysGpu(const T* x, const T* y, const T* z, KeyType* keys, size_t numKeys, const Box<T>& box, unsigned bx, unsigned by, unsigned bz)
{
    if (numKeys == 0) { return; }

    constexpr int threadsPerBlock = 256;
    computeSfcMixDKeysKernel<<<iceil(numKeys, threadsPerBlock), threadsPerBlock>>>(keys, x, y, z, numKeys, box, bx, by, bz);
    checkGpuErrors(cudaGetLastError());
}

template void
computeSfcKeysGpu(const float*, const float*, const float*, MortonKey<unsigned>*, size_t, const Box<float>&);
template void
computeSfcKeysGpu(const double*, const double*, const double*, MortonKey<unsigned>*, size_t, const Box<double>&);
template void
computeSfcKeysGpu(const float*, const float*, const float*, MortonKey<uint64_t>*, size_t, const Box<float>&);
template void
computeSfcKeysGpu(const double*, const double*, const double*, MortonKey<uint64_t>*, size_t, const Box<double>&);

template void
computeSfcKeysGpu(const float*, const float*, const float*, HilbertKey<unsigned>*, size_t, const Box<float>&);
template void
computeSfcKeysGpu(const double*, const double*, const double*, HilbertKey<unsigned>*, size_t, const Box<double>&);
template void
computeSfcKeysGpu(const float*, const float*, const float*, HilbertKey<uint64_t>*, size_t, const Box<float>&);
template void
computeSfcKeysGpu(const double*, const double*, const double*, HilbertKey<uint64_t>*, size_t, const Box<double>&);

template void
computeSfcMixDKeysGpu(const float*, const float*, const float*, HilbertMixDKey<unsigned>*, size_t, const Box<float>&, unsigned, unsigned, unsigned);
template void
computeSfcMixDKeysGpu(const double*, const double*, const double*, HilbertMixDKey<unsigned>*, size_t, const Box<double>&, unsigned, unsigned, unsigned);
template void
computeSfcMixDKeysGpu(const float*, const float*, const float*, HilbertMixDKey<uint64_t>*, size_t, const Box<float>&, unsigned, unsigned, unsigned);
template void
computeSfcMixDKeysGpu(const double*, const double*, const double*, HilbertMixDKey<uint64_t>*, size_t, const Box<double>&, unsigned, unsigned, unsigned);

} // namespace cstone
