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
#include "cstone/findneighbors.hpp"

#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"
#include "sph/hydro_ve/iad_kern.hpp"
#include "sph/hydro_ve/divv_curlv_kern.hpp"

namespace sph
{
namespace cuda
{

template<class Tc, class T, class KeyType>
__global__ void iadDivvCurlvGpu(T sincIndex, T K, unsigned ngmax, const cstone::Box<Tc> box, size_t first, size_t last,
                                size_t numParticles, const KeyType* particleKeys, const Tc* x, const Tc* y, const Tc* z,
                                const T* vx, const T* vy, const T* vz, const T* h, const T* wh, const T* whd,
                                const T* xm, const T* kx, T* c11, T* c12, T* c13, T* c22, T* c23, T* c33, T* divv,
                                T* curlv, T* dV11, T* dV12, T* dV13, T* dV22, T* dV23, T* dV33, bool doGradV)
{
    cstone::LocalIndex tid = blockDim.x * blockIdx.x + threadIdx.x;
    cstone::LocalIndex i   = tid + first;

    if (i >= last) return;

    // need to hard-code ngmax stack allocation for now
    assert(ngmax <= NGMAX && "ngmax too big, please increase NGMAX to desired size");
    cstone::LocalIndex neighbors[NGMAX];
    unsigned           neighborsCount;

    // starting from CUDA 11.3, dynamic stack allocation is available with the following command
    // int* neighbors = (int*)alloca(ngmax * sizeof(int));

    cstone::findNeighbors(i, x, y, z, h, box, cstone::sfcKindPointer(particleKeys), neighbors, &neighborsCount,
                          numParticles, ngmax);
    neighborsCount = stl::min(neighborsCount, ngmax);

    IADJLoop(i, sincIndex, K, box, neighbors, neighborsCount, x, y, z, h, wh, whd, xm, kx, c11, c12, c13, c22, c23,
             c33);
    divV_curlVJLoop(i, sincIndex, K, box, neighbors, neighborsCount, x, y, z, vx, vy, vz, h, c11, c12, c13, c22, c23,
                    c33, wh, whd, kx, xm, divv, curlv, dV11, dV12, dV13, dV22, dV23, dV33, doGradV);
}

template<class Dataset>
void computeIadDivvCurlv(size_t startIndex, size_t endIndex, unsigned ngmax, Dataset& d,
                         const cstone::Box<typename Dataset::RealType>& box)
{
    using T = typename Dataset::RealType;

    // number of locally present particles, including halos
    size_t sizeWithHalos       = d.x.size();
    size_t numParticlesCompute = endIndex - startIndex;

    unsigned numThreads = 128;
    unsigned numBlocks  = (numParticlesCompute + numThreads - 1) / numThreads;

    bool doGradV = d.devData.x.size() == d.devData.dV11.size();

    iadDivvCurlvGpu<<<numBlocks, numThreads>>>(
        d.sincIndex, d.K, ngmax, box, startIndex, endIndex, sizeWithHalos, rawPtr(d.devData.keys), rawPtr(d.devData.x),
        rawPtr(d.devData.y), rawPtr(d.devData.z), rawPtr(d.devData.vx), rawPtr(d.devData.vy), rawPtr(d.devData.vz),
        rawPtr(d.devData.h), rawPtr(d.devData.wh), rawPtr(d.devData.whd), rawPtr(d.devData.xm), rawPtr(d.devData.kx),
        rawPtr(d.devData.c11), rawPtr(d.devData.c12), rawPtr(d.devData.c13), rawPtr(d.devData.c22),
        rawPtr(d.devData.c23), rawPtr(d.devData.c33), rawPtr(d.devData.divv), rawPtr(d.devData.curlv),
        rawPtr(d.devData.dV11), rawPtr(d.devData.dV12), rawPtr(d.devData.dV13), rawPtr(d.devData.dV22),
        rawPtr(d.devData.dV23), rawPtr(d.devData.dV33), doGradV);
    checkGpuErrors(cudaDeviceSynchronize());
}

#define IAD_DIVV_CURLV(real, key)                                                                                      \
    template void computeIadDivvCurlv(size_t, size_t, unsigned, sphexa::ParticlesData<real, key, cstone::GpuTag>& d,   \
                                      const cstone::Box<real>&)

IAD_DIVV_CURLV(double, uint32_t);
IAD_DIVV_CURLV(double, uint64_t);
IAD_DIVV_CURLV(float, uint32_t);
IAD_DIVV_CURLV(float, uint64_t);

} // namespace cuda
} // namespace sph
