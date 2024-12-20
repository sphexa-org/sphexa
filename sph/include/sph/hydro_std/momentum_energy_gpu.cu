/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
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
 * @brief Pressure gradient (momentum) and energy i-loop GPU driver
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <cub/cub.cuh>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/findneighbors.hpp"
#include "cstone/traversal/find_neighbors.cuh"

#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"
#include "sph/util/device_math.cuh"
#include "sph/hydro_std/momentum_energy_kern.hpp"

namespace sph
{

using cstone::GpuConfig;
using cstone::LocalIndex;
using cstone::TravConfig;
using cstone::TreeNodeIndex;

static __device__ float minDt_device;

template<class Tc, class Tm, class T, class Tm1, class KeyType>
__global__ void cudaGradP(Tc K, Tc Kcour, unsigned ngmax, cstone::Box<Tc> box, const LocalIndex* grpStart,
                          const LocalIndex* grpEnd, LocalIndex numGroups, const cstone::OctreeNsView<Tc, KeyType> tree,
                          const Tc* x, const Tc* y, const Tc* z, const T* vx, const T* vy, const T* vz, const T* h,
                          const Tm* m, const T* rho, const T* p, const T* c, const T* c11, const T* c12, const T* c13,
                          const T* c22, const T* c23, const T* c33, const T* wh, const T* whd, T* grad_P_x, T* grad_P_y,
                          T* grad_P_z, Tm1* du, LocalIndex* nidx, TreeNodeIndex* globalPool, float* groupDt)
{
    unsigned laneIdx     = threadIdx.x & (GpuConfig::warpSize - 1);
    unsigned targetIdx   = 0;
    unsigned warpIdxGrid = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;

    LocalIndex* neighborsWarp = nidx + ngmax * TravConfig::targetSize * warpIdxGrid;

    T dt_i = INFINITY;

    while (true)
    {
        // first thread in warp grabs next target
        if (laneIdx == 0) { targetIdx = atomicAdd(&cstone::targetCounterGlob, 1); }
        targetIdx = cstone::shflSync(targetIdx, 0);

        if (targetIdx >= numGroups) { break; }

        LocalIndex bodyBegin = grpStart[targetIdx];
        LocalIndex bodyEnd   = grpEnd[targetIdx];
        LocalIndex i         = bodyBegin + laneIdx;

        auto ncTrue = traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, neighborsWarp, ngmax, globalPool);

        unsigned ncCapped = stl::min(ncTrue[0], ngmax);
        T        maxvsignal;

        if (i < bodyEnd)
        {
            momentumAndEnergyJLoop<TravConfig::targetSize>(i, K, box, neighborsWarp + laneIdx, ncCapped, x, y, z, vx,
                                                           vy, vz, h, m, rho, p, c, c11, c12, c13, c22, c23, c33, wh,
                                                           whd, grad_P_x, grad_P_y, grad_P_z, du, &maxvsignal);
        }

        auto dt_lane = (i < bodyEnd) ? tsKCourant(maxvsignal, h[i], c[i], Kcour) : INFINITY;
        if (groupDt != nullptr)
        {
            auto min_dt_group = cstone::warpMin(dt_lane);
            if (laneIdx == 0) { groupDt[targetIdx] = stl::min(groupDt[targetIdx], min_dt_group); }
        }

        dt_i = stl::min(dt_i, dt_lane);
    }

    typedef cub::BlockReduce<T, TravConfig::numThreads> BlockReduce;
    __shared__ typename BlockReduce::TempStorage        temp_storage;

    BlockReduce reduce(temp_storage);
    T           blockMin = reduce.Reduce(dt_i, cub::Min());
    __syncthreads();

    if (threadIdx.x == 0) { atomicMinFloat(&minDt_device, blockMin); }
}

template<class Dataset>
void computeMomentumEnergyStdGpu(const GroupView& grp, float* groupDt, Dataset& d,
                                 const cstone::Box<typename Dataset::RealType>& box)
{
    auto [traversalPool, nidxPool] = cstone::allocateNcStacks(d.devData.traversalStack, d.ngmax);
    cstone::resetTraversalCounters<<<1, 1>>>();

    float huge = 1e10;
    checkGpuErrors(cudaMemcpyToSymbol(minDt_device, &huge, sizeof(huge)));
    cstone::resetTraversalCounters<<<1, 1>>>();

    cudaGradP<<<TravConfig::numBlocks(), TravConfig::numThreads>>>(
        d.K, d.Kcour, d.ngmax, box, grp.groupStart, grp.groupEnd, grp.numGroups, d.treeView, rawPtr(d.devData.x),
        rawPtr(d.devData.y), rawPtr(d.devData.z), rawPtr(d.devData.vx), rawPtr(d.devData.vy), rawPtr(d.devData.vz),
        rawPtr(d.devData.h), rawPtr(d.devData.m), rawPtr(d.devData.rho), rawPtr(d.devData.p), rawPtr(d.devData.c),
        rawPtr(d.devData.c11), rawPtr(d.devData.c12), rawPtr(d.devData.c13), rawPtr(d.devData.c22),
        rawPtr(d.devData.c23), rawPtr(d.devData.c33), rawPtr(d.devData.wh), rawPtr(d.devData.whd), rawPtr(d.devData.ax),
        rawPtr(d.devData.ay), rawPtr(d.devData.az), rawPtr(d.devData.du), nidxPool, traversalPool, groupDt);

    checkGpuErrors(cudaGetLastError());

    float minDt;
    checkGpuErrors(cudaMemcpyFromSymbol(&minDt, minDt_device, sizeof(minDt)));
    d.minDtCourant = minDt;
}

template void computeMomentumEnergyStdGpu(const GroupView& grp, float*, sphexa::ParticlesData<cstone::GpuTag>& d,
                                          const cstone::Box<SphTypes::CoordinateType>&);
} // namespace sph
