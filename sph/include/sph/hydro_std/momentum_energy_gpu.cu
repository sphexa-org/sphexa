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

#include <limits>

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include "sph/neighborhood_gpu.hpp"
#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"
#include "sph/hydro_std/momentum_energy_kern.hpp"

namespace sph
{

using cstone::LocalIndex;

/*! @brief Mark particles with NaN acceleration for removal by setting neighbor counts to 0
 * @param[in]    grp   active particle groups
 * @param[inout] ax    x particle acceleration
 * @param[inout] ay
 * @param[inout] az
 * @param[inout] nc    neighbor counts
 */
template<class Ta, class Tu>
__global__ void markNaN(GroupView grp, Ta* ax, Ta* ay, Ta* az, Tu* du, unsigned* nc)
{
    LocalIndex laneIdx = threadIdx.x & (cstone::GpuConfig::warpSize - 1);
    LocalIndex warpIdx = (blockDim.x * blockIdx.x + threadIdx.x) >> cstone::GpuConfig::warpSizeLog2;
    if (warpIdx >= grp.numGroups) { return; }

    LocalIndex i = grp.groupStart[warpIdx] + laneIdx;
    if (i >= grp.groupEnd[warpIdx]) { return; }

    if (std::isnan(ax[i]) || std::isnan(ay[i]) || std::isnan(az[i]) || std::isnan(du[i]))
    {
        ax[i] = Ta(0);
        ay[i] = Ta(0);
        az[i] = Ta(0);
        du[i] = Tu(0);
        nc[i] = 1;
    }
}

template<class Dataset>
void computeMomentumEnergyStdGpu(const GroupView& grp, Dataset& d, const cstone::Box<typename Dataset::RealType>&)
{
    momentumAndEnergyIjLoop(d.devData.neighborhood, d.K, d.Kcour, rawPtr(d.devData.m), rawPtr(d.devData.rho),
                            rawPtr(d.devData.vx), rawPtr(d.devData.vy), rawPtr(d.devData.vz), rawPtr(d.devData.p),
                            rawPtr(d.devData.c), rawPtr(d.devData.c11), rawPtr(d.devData.c12), rawPtr(d.devData.c13),
                            rawPtr(d.devData.c22), rawPtr(d.devData.c23), rawPtr(d.devData.c33), rawPtr(d.devData.wh),
                            rawPtr(d.devData.du), rawPtr(d.devData.ax), rawPtr(d.devData.ay), rawPtr(d.devData.az),
                            rawPtr(d.devData.dtCourant));

    {
        unsigned numThreads       = 256;
        unsigned numWarpsPerBlock = numThreads / cstone::GpuConfig::warpSize;
        unsigned numBlocks        = (grp.numGroups + numWarpsPerBlock - 1) / numWarpsPerBlock;
        if (numBlocks > 0)
        {
            markNaN<<<numBlocks, numThreads>>>(grp, rawPtr(d.devData.ax), rawPtr(d.devData.ay), rawPtr(d.devData.az),
                                               rawPtr(d.devData.du), rawPtr(d.devData.nc));
        }
    }

    using DtCourantType = typename std::decay_t<decltype(d.devData.dtCourant)>::value_type;
    auto minDt          = thrust::reduce(thrust::device, rawPtr(d.devData.dtCourant) + grp.firstBody,
                                         rawPtr(d.devData.dtCourant) + grp.lastBody,
                                         std::numeric_limits<DtCourantType>::infinity(), thrust::minimum<DtCourantType>());
    d.minDtCourant      = minDt;
}

template void computeMomentumEnergyStdGpu(const GroupView& grp, sphexa::ParticlesData<cstone::GpuTag>& d,
                                          const cstone::Box<SphTypes::CoordinateType>&);
} // namespace sph
