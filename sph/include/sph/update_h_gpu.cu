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
#include "cstone/traversal/find_neighbors.cuh"

#include "sph/kernels.hpp"
#include "sph/particles_data.hpp"
#include "sph/sph_gpu.hpp"

namespace sph
{
using cstone::GpuConfig;
using cstone::LocalIndex;
using cstone::NcStats;
using cstone::TravConfig;
using cstone::TreeNodeIndex;

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

__device__ bool nc_h_convergenceFailure = false;

template<class Tc, class T, class KeyType>
__global__ void
updateSmoothingLengthIterativeGpuKernel(unsigned ng0, unsigned ngmax, const cstone::Box<Tc> box,
                                        const LocalIndex* grpStart, const LocalIndex* grpEnd, LocalIndex numGroups,
                                        const cstone::OctreeNsView<Tc, KeyType> tree, const Tc* x, const Tc* y,
                                        const Tc* z, T* h, unsigned* nc, LocalIndex* nidx, TreeNodeIndex* globalPool)
{
    unsigned laneIdx     = threadIdx.x & (GpuConfig::warpSize - 1);
    unsigned targetIdx   = 0;
    unsigned warpIdxGrid = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;

    LocalIndex* neighborsWarp = nidx + ngmax * TravConfig::targetSize * warpIdxGrid;

    while (true)
    {
        // first thread in warp grabs next target
        if (laneIdx == 0) { targetIdx = atomicAdd(&cstone::targetCounterGlob, 1); }
        targetIdx = cstone::shflSync(targetIdx, 0);

        if (targetIdx >= numGroups) return;

        LocalIndex bodyBegin = grpStart[targetIdx];
        LocalIndex bodyEnd   = grpEnd[targetIdx];
        LocalIndex i         = bodyBegin + laneIdx;

        unsigned ncSph =
            1 + traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, neighborsWarp, ngmax, globalPool)[0];

        constexpr int ncMaxIteration = 9;
        for (int ncIt = 0; ncIt <= ncMaxIteration; ++ncIt)
        {
            bool repeat = (ncSph < ng0 / 4 || (ncSph - 1) > ngmax) && i < bodyEnd;
            if (!cstone::ballotSync(repeat)) { break; }
            if (repeat) { h[i] = updateH(ng0, ncSph, h[i]); }
            ncSph =
                1 + traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, neighborsWarp, ngmax, globalPool)[0];

            if (ncIt == ncMaxIteration) { nc_h_convergenceFailure = true; }
        }

        if (i >= bodyEnd) continue;

        nc[i] = ncSph;
    }
}

template<class T, class Dataset>
void updateSmoothingLengthIterativeGpu(const cstone::GroupView& grp, Dataset& d, const cstone::Box<T>& box)
{
    auto [traversalPool, nidxPool] = cstone::allocateNcStacks(d.devData.traversalStack, d.ngmax);
    cstone::resetTraversalCounters<<<1, 1>>>();

    updateSmoothingLengthIterativeGpuKernel<<<TravConfig::numBlocks(), TravConfig::numThreads>>>(
        d.ng0, d.ngmax, box, grp.groupStart, grp.groupEnd, grp.numGroups, d.treeView, rawPtr(d.devData.x),
        rawPtr(d.devData.y), rawPtr(d.devData.z), rawPtr(d.devData.h), rawPtr(d.devData.nc), nidxPool, traversalPool);
    checkGpuErrors(cudaDeviceSynchronize());

    NcStats::type stats[NcStats::numStats];
    checkGpuErrors(cudaMemcpyFromSymbol(stats, GPU_SYMBOL(cstone::ncStats), NcStats::numStats * sizeof(NcStats::type)));

    bool convergenceFailure;
    checkGpuErrors(cudaMemcpyFromSymbol(&convergenceFailure, GPU_SYMBOL(nc_h_convergenceFailure), sizeof(bool)));

    NcStats::type maxP2P   = stats[cstone::NcStats::maxP2P];
    NcStats::type maxStack = stats[cstone::NcStats::maxStack];

    d.devData.stackUsedNc = maxStack;

    if (maxP2P == 0xFFFFFFFF) { throw std::runtime_error("GPU traversal stack exhausted in neighbor search\n"); }
    if (convergenceFailure) { throw std::runtime_error("coupled nc/h-updated failed to converge"); }
}

template void updateSmoothingLengthIterativeGpu(const cstone::GroupView&, sphexa::ParticlesData<cstone::GpuTag>&,
                                                const cstone::Box<SphTypes::CoordinateType>&);

} // namespace sph
