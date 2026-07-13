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

#include "sph/kernels.hpp"
#include "sph/particles_data.hpp"
#include "sph/sph_gpu.hpp"

namespace sph
{
using cstone::GpuConfig;
using cstone::LocalIndex;
using cstone::TreeNodeIndex;

__device__ bool nc_h_convergenceFailure = false;

template<class Th, class KeyType>
__global__ void updateSmoothingLengthGpuKernel(GroupView grp, unsigned ng0, const unsigned* nc, Th* h, KeyType* keys)
{
    LocalIndex laneIdx = threadIdx.x & (cstone::GpuConfig::warpSize - 1);
    LocalIndex warpIdx = (blockDim.x * blockIdx.x + threadIdx.x) >> cstone::GpuConfig::warpSizeLog2;
    if (warpIdx >= grp.numGroups) { return; }

    LocalIndex i = grp.groupStart[warpIdx] + laneIdx;
    if (i >= grp.groupEnd[warpIdx]) { return; }

    if (nc[i] <= 1)
    {
        keys[i]                 = cstone::removeKey<KeyType>{};
        nc_h_convergenceFailure = true;
    }
    h[i] = updateH(ng0, nc[i], h[i]);
}

template<class Th, class KeyType>
bool updateSmoothingLengthGpu(const GroupView& grp, unsigned ng0, const unsigned* nc, Th* h, KeyType* keys)
{
    unsigned numThreads       = 256;
    unsigned numWarpsPerBlock = numThreads / cstone::GpuConfig::warpSize;
    unsigned numBlocks        = (grp.numGroups + numWarpsPerBlock - 1) / numWarpsPerBlock;
    if (numBlocks == 0) { return false; }
    updateSmoothingLengthGpuKernel<<<numBlocks, numThreads>>>(grp, ng0, nc, h, keys);

    bool convergenceFailure;
    checkGpuErrors(cudaMemcpyFromSymbol(&convergenceFailure, GPU_SYMBOL(nc_h_convergenceFailure), sizeof(bool)));
    return convergenceFailure;
}

template bool updateSmoothingLengthGpu(const GroupView& grp, unsigned ng0, const unsigned* nc, float* h, uint64_t*);
template bool updateSmoothingLengthGpu(const GroupView& grp, unsigned ng0, const unsigned* nc, double* h, uint64_t*);

template<class Tc, class T, class KeyType>
__global__ __launch_bounds__(128) void updateSmoothingLengthIterativeGpuKernel(
    GroupView grp, unsigned ng0, unsigned ngmax, const cstone::Box<Tc> box,
    const cstone::OctreeNsView<Tc, KeyType> tree, const Tc* __restrict__ x, const Tc* __restrict__ y,
    const Tc* __restrict__ z, T* __restrict__ h, unsigned* __restrict__ nc)
{
    LocalIndex laneIdx = threadIdx.x & (cstone::GpuConfig::warpSize - 1);
    LocalIndex warpIdx = (blockDim.x * blockIdx.x + threadIdx.x) >> cstone::GpuConfig::warpSizeLog2;
    if (warpIdx >= grp.numGroups) { return; }

    const LocalIndex i = grp.groupStart[warpIdx] + laneIdx;
    if (i >= grp.groupEnd[warpIdx]) { return; }

    const unsigned ngmin = ng0 / 4;

    constexpr int maxIteration = 10;

    unsigned ncSph = 1 + findNeighbors(i, x, y, z, h, tree, box, ngmax);

    int iteration = 0;
    while ((ngmin > ncSph || (ncSph - 1) > ngmax) && iteration++ < maxIteration)
    {
        h[i]  = updateH(ng0, ncSph, h[i]);
        ncSph = 1 + findNeighbors(i, x, y, z, h, tree, box, ngmax);
    }
    if (iteration == maxIteration && (ngmin > ncSph || (ncSph - 1) > ngmax)) { ncSph = 1; }

    nc[i] = ncSph;
}

template<class T, class Dataset>
void updateSmoothingLengthIterativeGpu(const cstone::GroupView& grp, Dataset& d, const cstone::Box<T>& box)
{
    unsigned numThreads       = 128;
    unsigned numWarpsPerBlock = numThreads / cstone::GpuConfig::warpSize;
    unsigned numBlocks        = (grp.numGroups + numWarpsPerBlock - 1) / numWarpsPerBlock;
    if (numBlocks == 0) { return; }

    updateSmoothingLengthIterativeGpuKernel<<<numBlocks, numThreads>>>(
        grp, d.ng0, d.ngmax, box, d.treeView, rawPtr(d.x), rawPtr(d.y), rawPtr(d.z), rawPtr(d.h), rawPtr(d.nc));
}

template void updateSmoothingLengthIterativeGpu(const cstone::GroupView&,
                                                sphexa::ParticlesData<cstone::execution::Gpu>&,
                                                const cstone::Box<SphTypes::CoordinateType>&);

} // namespace sph
