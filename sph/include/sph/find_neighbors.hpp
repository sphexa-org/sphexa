#pragma once

#include "cstone/traversal/groups.hpp"

#include "sph/find_neighbors_gpu.hpp"
#include "sph/neighborhood.hpp"
#include "sph/particles_data.hpp"

namespace sph
{

inline void findNeighborsSfc(const cstone::GroupView& groups, sphexa::ParticlesData<cstone::CpuTag>& d,
                             const cstone::Box<SphTypes::CoordinateType>& box, bool subgroups = false)
{
    if (d.ng0 > d.ngmax) { throw std::runtime_error("ng0 should be smaller than ngmax\n"); }

    d.neighborhood.build(groups, d, box, subgroups);
}

} // namespace sph
