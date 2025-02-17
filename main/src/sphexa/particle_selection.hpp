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

using ParticleIdType = uint64_t; // TODO: retrieve type from ParticlesData
constexpr ParticleIdType msbMask = static_cast<ParticleIdType>(1) << (sizeof(ParticleIdType)*8 - 1);

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

// Identify tagged particles and save their indexes
extern void findSelectedParticlesIndexes_gpu(const ParticlesData<cstone::GpuTag>& d, std::vector<uint64_t>& localSelectedParticlesIndexes);

// TODO: search in size_t firstIndex, size_t lastIndex
template<class AccType>
void findSelectedParticlesIndexes(const ParticlesData<AccType>& d, std::vector<uint64_t>& localSelectedParticlesIndexes)
{
    if constexpr (cstone::HaveGpu<AccType>{})
    {
        findSelectedParticlesIndexes_gpu(d, localSelectedParticlesIndexes);
    }
    else
    {
        // Find the selected particles in local id list and save their indexes
        // TODO: switch to GPU-like implementation?
        uint64_t particleIndex = 0;
        std::for_each(d.id.begin(), d.id.end(), [&localSelectedParticlesIndexes, &particleIndex](auto& particleId){
            if((particleId & msbMask) != 0) {// check MSB
                localSelectedParticlesIndexes.push_back(particleIndex); // TODO: inefficient due to resizing, avoid push_back usage
            }
            particleIndex++;
        });
    }
}

}