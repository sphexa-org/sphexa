#include "sph/find_neighbors_gpu.hpp"
#include "sph/neighborhood_gpu.hpp"
#include "sph/particles_data.hpp"

namespace sph
{

template<class T, class Dataset>
void findNeighborsSfcGpu(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box)
{
    d.neighborhood = buildNeighborhoodGpu(groups, d, box);
}

template void findNeighborsSfcGpu(const cstone::GroupView&, sphexa::ParticlesData<cstone::GpuTag>&,
                                  const cstone::Box<SphTypes::CoordinateType>&);

} // namespace sph
