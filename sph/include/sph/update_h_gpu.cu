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

template<class Tc, class T, class KeyType>
__global__ __launch_bounds__(128) void updateSmoothingLengthIterativeGpuKernel(
    GroupView grp, unsigned ng0, unsigned ngmax, const cstone::Box<Tc> box,
    const cstone::OctreeNsView<Tc, KeyType> tree, const Tc* __restrict__ x, const Tc* __restrict__ y,
    const Tc* __restrict__ z, T* __restrict__ h, unsigned* __restrict__ nc, KeyType* __restrict__ keys)
{
    LocalIndex laneIdx = threadIdx.x & (cstone::GpuConfig::warpSize - 1);
    LocalIndex warpIdx = (blockDim.x * blockIdx.x + threadIdx.x) >> cstone::GpuConfig::warpSizeLog2;
    const LocalIndex i = grp.groupStart[warpIdx] + laneIdx;

    if (warpIdx >= grp.numGroups || i >= grp.groupEnd[warpIdx]) { return; }

    h[i] = updateH(ng0, nc[i], h[i]);

    updateHIterative(ng0, ngmax, box, tree, i, x, y, z, h, nc);

    if (nc[i] <= 1)
    {
        keys[i]                 = cstone::removeKey<KeyType>{};
        nc_h_convergenceFailure = true;
    }
}

template<class T, class Dataset>
bool updateSmoothingLengthIterativeGpu(const cstone::GroupView& grp, Dataset& d, const cstone::Box<T>& box)
{
    unsigned numThreads       = 128;
    unsigned numWarpsPerBlock = numThreads / cstone::GpuConfig::warpSize;
    unsigned numBlocks        = (grp.numGroups + numWarpsPerBlock - 1) / numWarpsPerBlock;
    if (numBlocks == 0) { return false; }

    updateSmoothingLengthIterativeGpuKernel<<<numBlocks, numThreads>>>(
        grp, d.ng0, d.ngmax, box, d.treeView, rawPtr(d.x), rawPtr(d.y), rawPtr(d.z), rawPtr(d.h), rawPtr(d.nc));

    bool convergenceFailure;
    checkGpuErrors(cudaMemcpyFromSymbol(&convergenceFailure, GPU_SYMBOL(nc_h_convergenceFailure), sizeof(bool)));
    return convergenceFailure;
}

template bool updateSmoothingLengthIterativeGpu(const cstone::GroupView&,
                                                sphexa::ParticlesData<cstone::execution::Gpu>&,
                                                const cstone::Box<SphTypes::CoordinateType>&);

} // namespace sph
