/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
 *               2022 University of Basel
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
 * @brief Smoothing length update on the GPU
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/cuda/cub.hpp"
#include "cstone/cuda/gpu_config.cuh"
#include "sph/sph_gpu.hpp"
#include "sph/kernels.hpp"

namespace sph
{
using cstone::LocalIndex;

static __device__ unsigned long long n_removed = 0;

template<unsigned numThreads, class Th, class KeyType>
__global__ void updateSmoothingLengthGpuKernel(GroupView grp, unsigned ng0, const unsigned* nc, Th* h, KeyType* keys)
{
    LocalIndex laneIdx = threadIdx.x & (cstone::GpuConfig::warpSize - 1);
    LocalIndex warpIdx = (blockDim.x * blockIdx.x + threadIdx.x) >> cstone::GpuConfig::warpSizeLog2;
    unsigned long long n_removed_local = 0;

    if (warpIdx < grp.numGroups)
    {

        LocalIndex i = grp.groupStart[warpIdx] + laneIdx;
        if (i < grp.groupEnd[warpIdx])
        {

            if (nc[i] <= 1)
            {
                keys[i]         = cstone::removeKey<KeyType>{};
                n_removed_local = 1;
            }
            h[i] = updateH(ng0, nc[i], h[i]);
        }
    }
    typedef cub::BlockReduce<unsigned long, numThreads> BlockReduce;
    __shared__ typename BlockReduce::TempStorage        temp_storage;
    BlockReduce                                         reduce(temp_storage);

    unsigned long long n_removed_block = reduce.Sum(n_removed_local);
    __syncthreads();
    if (threadIdx.x == 0) { atomicAdd(&n_removed, n_removed_block); }
}

template<class Th, class KeyType>
unsigned long long updateSmoothingLengthGpu(const GroupView& grp, unsigned ng0, const unsigned* nc, Th* h,
                                            KeyType* keys)
{
    constexpr unsigned numThreads       = 256;
    unsigned           numWarpsPerBlock = numThreads / cstone::GpuConfig::warpSize;
    unsigned           numBlocks        = (grp.numGroups + numWarpsPerBlock - 1) / numWarpsPerBlock;
    if (numBlocks == 0) { return false; }
    unsigned long long n_removed_host = 0;
    checkGpuErrors(cudaMemcpyToSymbol(GPU_SYMBOL(n_removed), &n_removed_host, sizeof(n_removed_host)));

    updateSmoothingLengthGpuKernel<numThreads><<<numBlocks, numThreads>>>(grp, ng0, nc, h, keys);
    checkGpuErrors(cudaDeviceSynchronize());
    checkGpuErrors(cudaGetLastError());

    checkGpuErrors(cudaMemcpyFromSymbol(&n_removed_host, GPU_SYMBOL(n_removed), sizeof(n_removed_host)));

    return n_removed_host;
}

template unsigned long long updateSmoothingLengthGpu(const GroupView& grp, unsigned ng0, const unsigned* nc, float* h,
                                                     uint64_t*);
template unsigned long long updateSmoothingLengthGpu(const GroupView& grp, unsigned ng0, const unsigned* nc, double* h,
                                                     uint64_t*);

} // namespace sph
