/*
 * MIT License
 *
 * SPH-EXA
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
 * @brief  CPU/GPU Particle subset positions identification functions
 *
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "sph/particles_data.hpp"

namespace sphexa
{

struct ParticleSelectionSphere
{
    std::array<double, 3> center;
    double radius;
};

// Find particles in user provided sphere and tag them
extern void findParticlesInSphere_gpu(ParticlesData<cstone::GpuTag>& d, size_t firstIndex, size_t lastIndex, const ParticleSelectionSphere& selSphereData);

template<class AccType>
void findParticlesInSphere(ParticlesData<AccType>& d, size_t firstIndex, size_t lastIndex, const ParticleSelectionSphere& selSphereData) {

    if constexpr (cstone::HaveGpu<AccType>{})
    {
        findParticlesInSphere_gpu(d, firstIndex, lastIndex, selSphereData);
    }
    else
    {
        // TODO: can we use C++23 zip iterators? Is there anything already implemented in SPH-EXA?
        const auto squareRadius = selSphereData.radius*selSphereData.radius;
//#pragma omp parallel for
        for(auto particleIndex = firstIndex; particleIndex < lastIndex; particleIndex++){
            auto x = d.x[particleIndex];
            auto y = d.y[particleIndex];
            auto z = d.z[particleIndex];
            if((x - selSphereData.center[0])*(x - selSphereData.center[0]) +
               (y - selSphereData.center[1])*(y - selSphereData.center[1]) +
               (z - selSphereData.center[2])*(z - selSphereData.center[2]) <= squareRadius) {
                d.id[particleIndex] = d.id[particleIndex] | msbMask;
            }
        }
    }
}


// Find particles in user provided id list and tag them
// TODO: only pass d.id
extern void findParticlesInIdList_gpu(ParticlesData<cstone::GpuTag>& d, size_t firstIndex, size_t lastIndex, const std::vector<ParticleIdType>& selParticlesIds);

template<class AccType>
void findParticlesInIdList(ParticlesData<AccType>& d, size_t firstIndex, size_t lastIndex, const std::vector<ParticleIdType>& selParticlesIds) {

    if constexpr (cstone::HaveGpu<AccType>{})
    {
        findParticlesInIdList_gpu(d, firstIndex, lastIndex, selParticlesIds);
    }
    else
    {
        const auto idListBeginIt = d.id.begin()+firstIndex;
        const auto idListEndIt = d.id.begin()+lastIndex;
        std::for_each(selParticlesIds.begin(), selParticlesIds.end(), [idListBeginIt, idListEndIt](auto selParticleId){
            auto lower = std::lower_bound(idListBeginIt, idListEndIt, selParticleId);
            if(lower != idListEndIt && *lower == selParticleId) {
                *lower = *lower | msbMask;
            }
        });
    }
}

}