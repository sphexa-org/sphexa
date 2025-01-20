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

namespace sph
{
namespace cuda
{

template<class Tt, class Tmass, class Trho, class Tp, class Tc>
__global__ void cudaEOS_HydroStdTemp(size_t firstParticle, size_t lastParticle, Tmass mui, Tt gamma, const Tt* temp,
                                     const Tmass* m, Trho* rho, Tp* p, Tc* c)
{
    unsigned i = firstParticle + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= lastParticle) return;

    util::tie(p[i], c[i]) = idealGasEOSTemp(temp[i], rho[i], mui, gamma);
}

template<class Tt, class Tmass, class Trho, class Tp, class Tc>
__global__ void cudaEOS_HydroStd_u(size_t firstParticle, size_t lastParticle, Tmass mui, Tt gamma, const Tt* u,
                                   const Tmass* m, Trho* rho, Tp* p, Tc* c)
{
    unsigned i = firstParticle + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= lastParticle) return;

    util::tie(p[i], c[i]) = idealGasEOS_u(u[i], rho[i], gamma);
}

template<class Tt, class Tmass, class Trho, class Tp, class Tc>
void computeEOS_HydroStd(size_t firstParticle, size_t lastParticle, Tmass mui, Tt gamma, const Tt* temp, const Tt* u,
                         const Tmass* m, Trho* rho, Tp* p, Tc* c)
{
    if (firstParticle == lastParticle) { return; }
    unsigned numThreads = 256;
    unsigned numBlocks  = cstone::iceil(lastParticle - firstParticle, numThreads);
    if (u == nullptr)
    {
        cudaEOS_HydroStdTemp<<<numBlocks, numThreads>>>(firstParticle, lastParticle, mui, gamma, temp, m, rho, p, c);
    }
    else { cudaEOS_HydroStd_u<<<numBlocks, numThreads>>>(firstParticle, lastParticle, mui, gamma, u, m, rho, p, c); }
    checkGpuErrors(cudaDeviceSynchronize());
}

template void computeEOS_HydroStd(size_t, size_t, double, double, const double*, const double*, const double*, double*,
                                  double*, double*);
template void computeEOS_HydroStd(size_t, size_t, float, double, const double*, const double*, const float*, float*,
                                  float*, float*);
template void computeEOS_HydroStd(size_t, size_t, float, float, const float*, const float*, const float*, float*,
                                  float*, float*);
} // namespace cuda
} // namespace sph
