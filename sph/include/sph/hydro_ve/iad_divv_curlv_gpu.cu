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

#include "sph/neighborhood_gpu.hpp"
#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"
#include "sph/hydro_ve/iad_divv_curlv_kern.hpp"

namespace sph
{
namespace gpu
{

template<class Dataset>
void computeIadDivvCurlv(const GroupView&, Dataset& d, const cstone::Box<typename Dataset::RealType>&)
{
    iadDivVCurlVIjLoop(d.neighborhood, d.K, rawPtr(d.vx), rawPtr(d.vy), rawPtr(d.vz), rawPtr(d.xm), rawPtr(d.kx),
                       rawPtr(d.nc), rawPtr(d.c11), rawPtr(d.c12), rawPtr(d.c13), rawPtr(d.c22), rawPtr(d.c23),
                       rawPtr(d.c33), rawPtr(d.wh), rawPtr(d.divv),
                       d.curlv.size() == d.x.size() ? rawPtr(d.curlv) : nullptr, rawPtr(d.dV11), rawPtr(d.dV12),
                       rawPtr(d.dV13), rawPtr(d.dV22), rawPtr(d.dV23), rawPtr(d.dV33), d.dV11.size() == d.x.size());

    checkGpuErrors(cudaDeviceSynchronize());
}

template void computeIadDivvCurlv(const GroupView& grp, sphexa::ParticlesData<cstone::GpuTag>& d,
                                  const cstone::Box<SphTypes::CoordinateType>&);

} // namespace gpu
} // namespace sph
