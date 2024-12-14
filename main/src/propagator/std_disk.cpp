/*! @file
 * @brief Translation unit for the disk propagator initializer
 *
 * @author Noah Kubli <noah.kubli@uzh.ch>
 */

#include "sph/types.hpp"
#include "propagator.h"
#include "std_disk.hpp"

namespace sphexa
{

template<class DomainType, class ParticleDataType>
std::unique_ptr<Propagator<DomainType, ParticleDataType>>
PropLib<DomainType, ParticleDataType>::makeDiskProp(std::ostream& output, size_t rank, const InitSettings& settings)
{
    return std::make_unique<DiskProp<DomainType, ParticleDataType>>(output, rank, settings);
}

#ifdef USE_CUDA
template struct PropLib<cstone::Domain<SphTypes::KeyType, SphTypes::CoordinateType, cstone::GpuTag>,
                        SimulationData<cstone::GpuTag>>;
#else
template struct PropLib<cstone::Domain<SphTypes::KeyType, SphTypes::CoordinateType, cstone::CpuTag>,
                        SimulationData<cstone::CpuTag>>;
#endif

} // namespace sphexa
