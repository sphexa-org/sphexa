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

#include <thrust/transform.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/traversal/find_neighbors.cuh"

#include "sph/neighborhood_gpu.hpp"
#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"
#include "sph/hydro_ve/xmass_kern.hpp"

namespace sph
{

using cstone::GpuConfig;
using cstone::LocalIndex;
using cstone::NcStats;
using cstone::TravConfig;
using cstone::TreeNodeIndex;

unsigned nsGroupSize() { return TravConfig::targetSize; }

namespace cuda
{

template<class Dataset>
void computeXMass(const GroupView& grp, Dataset& d, const cstone::Box<typename Dataset::RealType>& box)
{
    xmassIjLoop(getNeighborhoodGpu(d), d.K, rawPtr(d.devData.m), rawPtr(d.devData.wh), rawPtr(d.devData.xm));
}

template void computeXMass(const GroupView& grp, sphexa::ParticlesData<cstone::GpuTag>& d,
                           const cstone::Box<SphTypes::CoordinateType>&);

template<class Dataset>
void computeDensity(const GroupView& grp, Dataset& d, const cstone::Box<typename Dataset::RealType>& box)
{
    densityIjLoop(getNeighborhoodGpu(d), d.K, rawPtr(d.devData.m), rawPtr(d.devData.wh), rawPtr(d.devData.rho));
}

template void computeDensity(const GroupView&, sphexa::ParticlesData<cstone::GpuTag>& d,
                             const cstone::Box<SphTypes::CoordinateType>&);

} // namespace cuda
} // namespace sph
