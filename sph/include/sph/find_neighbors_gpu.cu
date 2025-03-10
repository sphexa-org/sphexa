#include "cstone/traversal/ijloop/gpu_alwaystraverse.cuh"

#include "sph/find_neighbors_gpu.hpp"
#include "sph/particles_data.hpp"

namespace sph
{

template<class T, class Dataset>
void findNeighborsSfcGpu(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box)
{
    d.neighborhood = cstone::ijloop::GpuAlwaysTraverseNeighborhood{d.ngmax}.build(
        d.treeView, box, d.devData.size(), groups, rawPtr(d.devData.x), rawPtr(d.devData.y), rawPtr(d.devData.z),
        rawPtr(d.devData.h));
}

template void findNeighborsSfcGpu(const cstone::GroupView&, sphexa::ParticlesData<cstone::GpuTag>&,
                                  const cstone::Box<SphTypes::CoordinateType>&);

} // namespace sph
