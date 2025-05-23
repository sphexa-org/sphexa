/*
 * MIT License
 *
 * Copyright (c) 2022 Politechnical University of Catalonia UPC
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
 * @brief  Implementation of stirring accelerations on GPUs
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/cuda/gpu_config.cuh"
#include "cstone/traversal/groups.hpp"

#include "sph/hydro_turb/stirring.hpp"

namespace sph
{
using cstone::GpuConfig;
using cstone::GroupView;
using cstone::LocalIndex;

template<class Tc, class Ta, class T>
__global__ void computeStirringKernel(GroupView grp, size_t numDim, const Tc* x, const Tc* y, const Tc* z, Ta* ax,
                                      Ta* ay, Ta* az, size_t numModes, const T* modes, const T* phaseReal,
                                      const T* phaseImag, const T* amplitudes, T solWeightNorm)
{
    LocalIndex laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
    LocalIndex warpIdx = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;
    if (warpIdx >= grp.numGroups) { return; }

    LocalIndex i = grp.groupStart[warpIdx] + laneIdx;
    if (i >= grp.groupEnd[warpIdx]) { return; }

    auto [turbAx, turbAy, turbAz] =
        stirParticle<Tc, Ta, T>(numDim, x[i], y[i], z[i], numModes, modes, phaseReal, phaseImag, amplitudes);

    ax[i] += solWeightNorm * turbAx;
    ay[i] += solWeightNorm * turbAy;
    az[i] += solWeightNorm * turbAz;
}

//! @brief Add stirring accelerations on the GPU, see CPU version for documentation
template<class Tc, class Ta, class T>
void computeStirringGpu(GroupView grp, size_t numDim, const Tc* x, const Tc* y, const Tc* z, Ta* ax, Ta* ay, Ta* az,
                        size_t numModes, const T* modes, const T* st_aka, const T* st_akb, const T* amplitudes,
                        T solWeightNorm)
{
    unsigned numThreads       = 256;
    unsigned numWarpsPerBlock = numThreads / GpuConfig::warpSize;
    unsigned numBlocks        = (grp.numGroups + numWarpsPerBlock - 1) / numWarpsPerBlock;

    if (numBlocks == 0) { return; }
    computeStirringKernel<<<numBlocks, numThreads>>>(grp, numDim, x, y, z, ax, ay, az, numModes, modes, st_aka, st_akb,
                                                     amplitudes, solWeightNorm);
}

// all double
template void computeStirringGpu(GroupView, size_t, const double*, const double*, const double*, double*, double*,
                                 double*, size_t, const double*, const double*, const double*, const double*, double);

// accelerations in single
template void computeStirringGpu(GroupView, size_t, const double*, const double*, const double*, float*, float*, float*,
                                 size_t, const double*, const double*, const double*, const double*, double);

// accelerations and modes in single
template void computeStirringGpu(GroupView, size_t, const double*, const double*, const double*, float*, float*, float*,
                                 size_t, const float*, const float*, const float*, const float*, float);

// all single
template void computeStirringGpu(GroupView, size_t, const float*, const float*, const float*, float*, float*, float*,
                                 size_t, const float*, const float*, const float*, const float*, float);

} // namespace sph
