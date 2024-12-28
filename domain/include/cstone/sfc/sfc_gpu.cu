/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
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
void computeSfcKeysGpu(const T* x, const T* y, const T* z, KeyType* keys, size_t numKeys, const Box<T>& box)
{
    if (numKeys == 0) { return; }

    constexpr int threadsPerBlock = 256;
    computeSfcKeysKernel<<<iceil(numKeys, threadsPerBlock), threadsPerBlock>>>(keys, x, y, z, numKeys, box);
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

} // namespace cstone
