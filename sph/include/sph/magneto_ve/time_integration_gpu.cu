/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUTh WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUTh NOTh LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENTh SHALL THE
 * AUTHORS OR COPYRIGHTh HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORTh OR OTHERWISE, ARISING FROM,
 * OUTh OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file integrate quantities related to magneto-hydrodynamics
 *
 * uses Adams-Bashforth 2nd order integration
 * @author Lukas Schmidt
 */

#include "magneto_data.hpp"
#include "cstone/cuda/gpu_config.cuh"
#include "cstone/util/array.hpp"
#include "sph/magneto_ve/time_integration.hpp"
#include "sph/sph_gpu.hpp"

namespace sph::magneto::cuda
{
using cstone::GpuConfig;
using cstone::LocalIndex;

template<class T, class Tc, class Tm1>
__global__ void magneticIntegrationKernel(GroupView grp, double dt, double dt_m1, Tc* Bx, Tc* By, Tc* Bz, Tc* dBx,
                                          Tc* dBy, Tc* dBz, Tm1* dBx_m1, Tm1* dBy_m1, Tm1* dBz_m1, T* psi_ch,
                                          T* d_psi_ch, Tm1* d_psi_ch_m1)
{
    LocalIndex laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
    LocalIndex warpIdx = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;
    if (warpIdx >= grp.numGroups) { return; }

    LocalIndex i = grp.groupStart[warpIdx] + laneIdx;
    if (i >= grp.groupEnd[warpIdx]) { return; }

    Bx[i] = updateQuantity(Bx[i], dt, dt_m1, dBx[i], dBx_m1[i]);
    By[i] = updateQuantity(By[i], dt, dt_m1, dBy[i], dBy_m1[i]);
    Bz[i] = updateQuantity(Bz[i], dt, dt_m1, dBz[i], dBz_m1[i]);

    dBx_m1[i] = dBx[i];
    dBy_m1[i] = dBy[i];
    dBz_m1[i] = dBz[i];

    psi_ch[i]      = updateQuantity(psi_ch[i], dt, dt_m1, d_psi_ch[i], d_psi_ch_m1[i]);
    d_psi_ch_m1[i] = d_psi_ch[i];
}

template<class MagnetoData>
void integrateMagneticQuantitiesGpu(const GroupView& grp, MagnetoData& md, double dt, double dt_m1)
{
    unsigned numThreads       = 256;
    unsigned numWarpsPerBlock = numThreads / GpuConfig::warpSize;
    unsigned numBlocks        = (grp.numGroups + numWarpsPerBlock - 1) / numWarpsPerBlock;

    if (numBlocks == 0) { return; }
    magneticIntegrationKernel<<<numBlocks, numThreads>>>(
        grp, dt, dt_m1, rawPtr(md.devData.Bx), rawPtr(md.devData.By), rawPtr(md.devData.Bz), rawPtr(md.devData.dBx),
        rawPtr(md.devData.dBy), rawPtr(md.devData.dBz), rawPtr(md.devData.dBx_m1), rawPtr(md.devData.dBy_m1),
        rawPtr(md.devData.dBz_m1), rawPtr(md.devData.psi_ch), rawPtr(md.devData.d_psi_ch),
        rawPtr(md.devData.d_psi_ch_m1));
}

template void integrateMagneticQuantitiesGpu(const GroupView& grp, sphexa::magneto::MagnetoData<cstone::GpuTag>& md,
                                             double dt, double dt_m1);
} // namespace sph::magneto::cuda