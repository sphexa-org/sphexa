/*! @file
 * @brief  CPU/GPU Particle subset positions identification function implementations
 *
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include "sph/particles_data.hpp"
#include "sph/particles_data_gpu.cuh"

#include "particle_selection.hpp"

namespace sphexa
{


// TODO: retrieve particle id type from ParticlesData
struct MaskFunctor
{
    __device__
    uint64_t operator()(uint64_t elId) const
    {
        return (elId & msbMask) != 0;
    }
};

void findParticlesInSphere_gpu(ParticlesData<cstone::GpuTag>& d, size_t firstIndex, size_t lastIndex, const ParticleSelectionSphere& selSphereData) {

}

void findParticlesInIdList_gpu(ParticlesData<cstone::GpuTag>& d, size_t firstIndex, size_t lastIndex, const std::vector<ParticleIdType>& selParticlesIds) {

}


// TODO: remove, debug only
//#include <inttypes.h>
// struct printf_functor
// {
//   __host__ __device__
//   void operator()(uint64_t x)
//   {
//     // note that using printf in a __device__ function requires
//     // code compiled for a GPU with compute capability 2.0 or
//     // higher (nvcc --arch=sm_20)
//     printf("%" PRIu64 "\n", x);
//   }
// };

}