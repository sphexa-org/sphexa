#include "sph/find_neighbors_gpu.hpp"

namespace sph
{

void findNeighborsSfc(const cstone::GroupView& groups, sphexa::ParticlesData<cstone::GpuTag>& d,
                      const cstone::Box<SphTypes::CoordinateType>& box, bool symmetric, bool clustered)
{
    if (d.ng0 > d.ngmax) { throw std::runtime_error("ng0 should be smaller than ngmax\n"); }

    d.devData.neighborhood.build(groups, d, box, symmetric, clustered);
}

} // namespace sph
