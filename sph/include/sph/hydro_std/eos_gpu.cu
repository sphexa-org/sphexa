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

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/util/tuple.hpp"

#include "sph/sph_gpu.hpp"
#include "sph/eos.hpp"
#include "sph/particles_data.hpp"

namespace sph
{
namespace cuda
{

template<class Tt, class Tm, class Thydro>
__global__ void cudaComputeIdealGasEOS_HydroStd(size_t firstParticle, size_t lastParticle, Tm mui, Tt gamma,
                                                const Tt* temp, const Tt* u, const Tm* m, Thydro* rho, Thydro* p,
                                                Thydro* c)
{
    unsigned i = firstParticle + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= lastParticle) return;

    if (u == nullptr) { util::tie(p[i], c[i]) = idealGasEOS(temp[i], rho[i], mui, gamma); }
    else { util::tie(p[i], c[i]) = idealGasEOS_u(u[i], rho[i], gamma); }
}

template<class Dataset>
void computeIdealGasEOS_HydroStd(size_t firstParticle, size_t lastParticle, Dataset& d)
{
    if (firstParticle == lastParticle) { return; }
    unsigned numThreads = 256;
    unsigned numBlocks  = cstone::iceil(lastParticle - firstParticle, numThreads);

    cudaComputeIdealGasEOS_HydroStd<<<numBlocks, numThreads>>>(
        firstParticle, lastParticle, d.muiConst, d.gamma, rawPtr(d.devData.temp), rawPtr(d.devData.u),
        rawPtr(d.devData.m), rawPtr(d.devData.rho), rawPtr(d.devData.p), rawPtr(d.devData.c));

    checkGpuErrors(cudaDeviceSynchronize());
}

template void computeIdealGasEOS_HydroStd(size_t, size_t, sphexa::ParticlesData<cstone::GpuTag>&);

template<typename Th, typename Tu>
__global__ void cudaComputeIsothermalEOS_HydroStd(size_t first, size_t last, Th cConst, Th* c, Th* rho, Th* p, Tu* temp)
{
    unsigned i = first + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= last) return;

    p[i] = isothermalEOS(cConst, rho[i]);
    c[i] = cConst;
    if (temp) { temp[i] = 0; }
}

template<typename Dataset>
void computeIsothermalEOS_HydroStd(size_t first, size_t last, Dataset& d)
{
    if (first == last) { return; }
    unsigned numThreads = 256;
    unsigned numBlocks  = cstone::iceil(last - first, numThreads);

    cudaComputeIsothermalEOS_HydroStd<<<numBlocks, numThreads>>>(first, last, d.soundSpeedConst, rawPtr(d.devData.c),
                                                                 rawPtr(d.devData.rho), rawPtr(d.devData.p),
                                                                 rawPtr(d.devData.temp));

    checkGpuErrors(cudaDeviceSynchronize());
}

template void computeIsothermalEOS_HydroStd(size_t, size_t, sphexa::ParticlesData<cstone::GpuTag>& d);

template<typename Th, typename Tt>
__global__ void cudaComputePolytropicEOS_HydroStd(size_t first, size_t last, Tt polytropic_const, Tt polytropic_index,
                                                  Th* rho, Th* p, Tt* temp, Th* c)
{
    unsigned i = first + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= last) return;

    util::tie(p[i], c[i]) = polytropicEOS(polytropic_const, polytropic_index, rho[i]);
    if (temp) { temp[i] = 0; }
}

template<typename Dataset>
void computePolytropicEOS_HydroStd(size_t first, size_t last, Dataset& d)
{
    if (first == last) { return; }
    unsigned numThreads = 256;
    unsigned numBlocks  = cstone::iceil(last - first, numThreads);

    cudaComputePolytropicEOS_HydroStd<<<numBlocks, numThreads>>>(first, last, d.polytropic_const, d.polytropic_index,
                                                                 rawPtr(d.devData.rho), rawPtr(d.devData.p),
                                                                 rawPtr(d.devData.temp), rawPtr(d.devData.c));

    checkGpuErrors(cudaDeviceSynchronize());
}

template void computePolytropicEOS_HydroStd(size_t, size_t, sphexa::ParticlesData<cstone::GpuTag>&);

} // namespace cuda
} // namespace sph
