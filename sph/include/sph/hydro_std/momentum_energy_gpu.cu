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

#include "sph/neighborhood_gpu.hpp"
#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"
#include "sph/hydro_std/momentum_energy_kern.hpp"

namespace sph
{

using cstone::LocalIndex;

template<class Dataset>
void computeMomentumEnergyStdGpu(Dataset& d, const cstone::Box<typename Dataset::RealType>&)
{
    d.minDtCourant = momentumAndEnergyIjLoop(
        d.neighborhood, d.K, d.Kcour, rawPtr(d.m), rawPtr(d.rho), rawPtr(d.nc), rawPtr(d.vx), rawPtr(d.vy),
        rawPtr(d.vz), rawPtr(d.p), rawPtr(d.c), rawPtr(d.c11), rawPtr(d.c12), rawPtr(d.c13), rawPtr(d.c22),
        rawPtr(d.c23), rawPtr(d.c33), rawPtr(d.wh), rawPtr(d.du), rawPtr(d.ax), rawPtr(d.ay), rawPtr(d.az));
}

template void computeMomentumEnergyStdGpu(sphexa::ParticlesData<cstone::execution::Gpu>& d,
                                          const cstone::Box<SphTypes::CoordinateType>&);

template<typename Thydro, typename T>
__global__ void relaxSystemKernel(size_t first, size_t last, Thydro* ax, Thydro* ay, Thydro* az, Thydro* vx, Thydro* vy,
                                  Thydro* vz, T relaxationTimescale)
{
    cstone::LocalIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= last) { return; }

    ax[i] -= vx[i] / relaxationTimescale;
    ay[i] -= vy[i] / relaxationTimescale;
    az[i] -= vz[i] / relaxationTimescale;
}

template<typename Thydro, typename T>
void relaxSystemGPU(size_t first, size_t last, Thydro* ax, Thydro* ay, Thydro* az, Thydro* vx, Thydro* vy, Thydro* vz,
                    T relaxationTimescale)
{
    cstone::LocalIndex numParticles = last - first;
    unsigned           numThreads   = 256;
    unsigned           numBlocks    = (numParticles + numThreads - 1) / numThreads;

    relaxSystemKernel<<<numBlocks, numThreads>>>(first, last, ax, ay, az, vx, vy, vz, relaxationTimescale);
    checkGpuErrors(cudaDeviceSynchronize());
}

#define RELAX_SYSTEM_GPU(Thydro, T)                                                                                    \
    template void relaxSystemGPU(size_t first, size_t last, Thydro* ax, Thydro* ay, Thydro* az, Thydro* vx,            \
                                 Thydro* vy, Thydro* vz, T relaxationTimescale);
RELAX_SYSTEM_GPU(float, double);
RELAX_SYSTEM_GPU(double, double);

} // namespace sph
