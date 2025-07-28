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
#include "cstone/traversal/find_neighbors.cuh"

#include "sph/neighborhood_gpu.hpp"
#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"
#include "sph/hydro_ve/av_switches_kern.hpp"

namespace sph::gpu
{

template<class Dataset>
void computeAVswitches(const GroupView&, Dataset& d, const cstone::Box<typename Dataset::RealType>&)
{
    // alpha is an input and output field, thus first copy alpha to a temporary vector to properly support symmetric
    // neighborhoods
    auto& tmp = d.devData.ax;
    assert(d.devData.ax.size() >= d.devData.alpha.size());
    checkGpuErrors(cudaMemcpyAsync(rawPtr(tmp), rawPtr(d.devData.alpha),
                                   sizeof(typename Dataset::HydroType) * d.devData.alpha.size(),
                                   cudaMemcpyDeviceToDevice));
    AVswitchesIjLoop(d.devData.neighborhood, d.K, d.minDt, d.alphamin, d.alphamax, d.decay_constant,
                     rawPtr(d.devData.xm), rawPtr(d.devData.kx), rawPtr(d.devData.divv), rawPtr(tmp),
                     rawPtr(d.devData.vx), rawPtr(d.devData.vy), rawPtr(d.devData.vz), rawPtr(d.devData.c),
                     rawPtr(d.devData.c11), rawPtr(d.devData.c12), rawPtr(d.devData.c13), rawPtr(d.devData.c22),
                     rawPtr(d.devData.c23), rawPtr(d.devData.c33), rawPtr(d.devData.wh), rawPtr(d.devData.alpha));

    checkGpuErrors(cudaDeviceSynchronize());
}

template void computeAVswitches(const GroupView& grp, sphexa::ParticlesData<cstone::GpuTag>& d,
                                const cstone::Box<SphTypes::CoordinateType>&);

} // namespace sph::gpu
