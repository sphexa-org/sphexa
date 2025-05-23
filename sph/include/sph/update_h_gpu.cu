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

#include "cstone/cuda/gpu_config.cuh"
#include "sph/sph_gpu.hpp"
#include "sph/kernels.hpp"

namespace sph
{
using cstone::LocalIndex;

template<class Th>
__global__ void updateSmoothingLengthGpuKernel(GroupView grp, unsigned ng0, const unsigned* nc, Th* h)
{
    LocalIndex laneIdx = threadIdx.x & (cstone::GpuConfig::warpSize - 1);
    LocalIndex warpIdx = (blockDim.x * blockIdx.x + threadIdx.x) >> cstone::GpuConfig::warpSizeLog2;
    if (warpIdx >= grp.numGroups) { return; }

    LocalIndex i = grp.groupStart[warpIdx] + laneIdx;
    if (i >= grp.groupEnd[warpIdx]) { return; }

    h[i] = updateH(ng0, nc[i], h[i]);
}

template<class Th>
void updateSmoothingLengthGpu(const GroupView& grp, unsigned ng0, const unsigned* nc, Th* h)
{
    unsigned numThreads       = 256;
    unsigned numWarpsPerBlock = numThreads / cstone::GpuConfig::warpSize;
    unsigned numBlocks        = (grp.numGroups + numWarpsPerBlock - 1) / numWarpsPerBlock;
    if (numBlocks == 0) { return; }
    updateSmoothingLengthGpuKernel<<<numBlocks, numThreads>>>(grp, ng0, nc, h);
}

template void updateSmoothingLengthGpu(const GroupView& grp, unsigned ng0, const unsigned* nc, float* h);
template void updateSmoothingLengthGpu(const GroupView& grp, unsigned ng0, const unsigned* nc, double* h);

} // namespace sph