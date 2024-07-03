/*
 * MIT License
 *
 * Copyright (c) 2023 CSCS, ETH Zurich
 *               2023 University of Basel
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
 * @brief additional fields GPU i-loop driver
 *
 * @author Lukas Schmidt
 */

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/findneighbors.hpp"
#include "cstone/traversal/find_neighbors.cuh"

#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"
#include "sph/hydro_ve/additional_fields_kern.hpp"

namespace sph
{
namespace cuda
{

using cstone::GpuConfig;
using cstone::LocalIndex;
using cstone::TravConfig;
using cstone::TreeNodeIndex;

template<class T, class Tm, class Tc, class KeyType>
__global__ void markRampGPU(const cstone::Box<Tc> box, size_t first, size_t last, unsigned ngmax,
                            const cstone::OctreeNsView<Tc, KeyType> tree, const Tc* x, const Tc* y, const Tc* z,
                            const T* h, const T* kx, const T* xm, const Tm* m, T* markRamp, T Atmin, T Atmax, T ramp,
                            LocalIndex* nidx, TreeNodeIndex* globalPool)
{
    unsigned laneIdx     = threadIdx.x & (GpuConfig::warpSize - 1);
    unsigned numTargets  = (last - first - 1) / TravConfig::targetSize + 1;
    unsigned targetIdx   = 0;
    unsigned warpIdxGrid = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;

    cstone::LocalIndex* neighborsWarp = nidx + ngmax * TravConfig::targetSize * warpIdxGrid;

    while (true)
    {
        // first thread in warp grabs next target
        if (laneIdx == 0) { targetIdx = atomicAdd(&cstone::targetCounterGlob, 1); }
        targetIdx = cstone::shflSync(targetIdx, 0);

        if (targetIdx >= numTargets) return;

        cstone::LocalIndex bodyBegin = first + targetIdx * TravConfig::targetSize;
        cstone::LocalIndex bodyEnd   = cstone::imin(bodyBegin + TravConfig::targetSize, last);
        cstone::LocalIndex i         = bodyBegin + laneIdx;

        auto ncTrue = traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, neighborsWarp, ngmax, globalPool);

        if (i >= last) continue;

        unsigned ncCapped = stl::min(ncTrue[0], ngmax);
        markRampJLoop<TravConfig::targetSize>(i, neighborsWarp + laneIdx, ncCapped, Atmin, Atmax, ramp, kx, xm, m,
                                              markRamp);
    }
}

template<class Dataset>
void computeMarkRamp(size_t first, size_t last, Dataset& d, const cstone::Box<typename Dataset::RealType>& box)
{
    unsigned numBodies = first - last;
    unsigned numBlocks = TravConfig::numBlocks(numBodies);

    auto [traversalPool, nidxPool] = cstone::allocateNcStacks(d.devData.traversalStack, numBodies, d.ngmax);
    cstone::resetTraversalCounters<<<1, 1>>>();
    markRampGPU<<<numBlocks, TravConfig::numThreads>>>(
        box, first, last, d.ngmax, d.treeView.nsView(), rawPtr(d.devData.x), rawPtr(d.devData.y), rawPtr(d.devData.z),
        rawPtr(d.devData.h), rawPtr(d.devData.kx), rawPtr(d.devData.xm), rawPtr(d.devData.m),
        rawPtr(d.devData.markRamp), d.Atmin, d.Atmax, d.ramp, nidxPool, traversalPool);
    checkGpuErrors(cudaDeviceSynchronize());
}

template void computeMarkRamp(size_t, size_t, sphexa::ParticlesData<cstone::GpuTag>& d, const cstone::Box<double>&);
template void computeMarkRamp(size_t, size_t, sphexa::ParticlesData<cstone::GpuTag>& d, const cstone::Box<float>&);

} // namespace cuda
} // namespace sph
