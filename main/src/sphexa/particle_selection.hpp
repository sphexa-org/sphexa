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

extern void findSelectedParticlesIndexes_gpu(const ParticlesData<cstone::GpuTag>& d, size_t first, size_t last, 
                                             std::vector<uint64_t>& localSelectedParticlesIndexes);

template<class AccType>
void findSelectedParticlesIndexes(const ParticlesData<AccType>& d, size_t first, size_t last, 
                                  std::vector<uint64_t>& localSelectedParticlesIndexes)
{
    if constexpr (cstone::HaveGpu<AccType>{})
    {
        findSelectedParticlesIndexes_gpu(d, first, last, localSelectedParticlesIndexes);
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