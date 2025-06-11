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
 * @brief Density i-loop GPU driver
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <limits>

#include "cstone/primitives/warpscan.cuh"

#include "sph/neighborhood_gpu.hpp"
#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"
#include "sph/hydro_ve/momentum_energy_kern.hpp"

namespace sph
{
namespace gpu
{

using cstone::GpuConfig;
using cstone::LocalIndex;

static __device__ float minDt_ve_device;

template<class T>
__global__ void reduceDt(const LocalIndex* __restrict__ grpStart, const LocalIndex* __restrict__ grpEnd,
                         const LocalIndex numGroups, const T* dtCourant, float* __restrict__ groupDt)
{
    unsigned laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
    unsigned grpIdx  = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;

    if (grpIdx >= numGroups) return;

    LocalIndex bodyBegin = grpStart[grpIdx];
    LocalIndex bodyEnd   = grpEnd[grpIdx];
    LocalIndex i         = bodyBegin + laneIdx;

    __shared__ float minBlockDt;
    if (threadIdx.x == 0) minBlockDt = std::numeric_limits<float>::infinity();
    __syncthreads();

    float dt         = i < bodyEnd ? dtCourant[i] : std::numeric_limits<T>::infinity();
    float minGroupDt = cstone::warpMin(dt);
    if (groupDt && laneIdx == 0) groupDt[grpIdx] = std::min(groupDt[grpIdx], minGroupDt);

    if (laneIdx == 0) cstone::atomicMinFloat(&minBlockDt, minGroupDt);
    __syncthreads();
    if (threadIdx.x == 0) cstone::atomicMinFloat(&minDt_ve_device, minBlockDt);
}

template<bool avClean, class Dataset>
void computeMomentumEnergy(const GroupView& grp, float* groupDt, Dataset& d,
                           const cstone::Box<typename Dataset::RealType>&)
{
    momentumAndEnergyIjLoop<avClean>(
        d.devData.neighborhood, d.K, d.Kcour, d.Atmin, d.Atmax, d.ramp, rawPtr(d.devData.vx), rawPtr(d.devData.vy),
        rawPtr(d.devData.vz), rawPtr(d.devData.m), rawPtr(d.devData.c), rawPtr(d.devData.kx), rawPtr(d.devData.alpha),
        rawPtr(d.devData.xm), rawPtr(d.devData.prho), rawPtr(d.devData.c11), rawPtr(d.devData.c12),
        rawPtr(d.devData.c13), rawPtr(d.devData.c22), rawPtr(d.devData.c23), rawPtr(d.devData.c33),
        rawPtr(d.devData.nc), rawPtr(d.devData.dV11), rawPtr(d.devData.dV12), rawPtr(d.devData.dV13),
        rawPtr(d.devData.dV22), rawPtr(d.devData.dV23), rawPtr(d.devData.dV33), rawPtr(d.devData.tdpdTrho),
        rawPtr(d.devData.wh), rawPtr(d.devData.du), rawPtr(d.devData.ax), rawPtr(d.devData.ay), rawPtr(d.devData.az),
        rawPtr(d.devData.dtCourant));

    float minDt = std::numeric_limits<float>::infinity();
    checkGpuErrors(cudaMemcpyToSymbolAsync(GPU_SYMBOL(minDt_ve_device), &minDt, sizeof(minDt)));

    constexpr LocalIndex threads = 256;
    const LocalIndex     blocks  = cstone::iceil(grp.numGroups, threads / GpuConfig::warpSize);
    reduceDt<<<blocks, threads>>>(grp.groupStart, grp.groupEnd, grp.numGroups, rawPtr(d.devData.dtCourant), groupDt);

    checkGpuErrors(cudaMemcpyFromSymbol(&minDt, GPU_SYMBOL(minDt_ve_device), sizeof(minDt)));
    d.minDtCourant = minDt;
}

#define MOM_ENERGY(avc)                                                                                                \
    template void computeMomentumEnergy<avc>(const GroupView& grp, float*, sphexa::ParticlesData<cstone::GpuTag>& d,   \
                                             const cstone::Box<SphTypes::CoordinateType>&)

MOM_ENERGY(true);
MOM_ENERGY(false);

} // namespace gpu
} // namespace sph
