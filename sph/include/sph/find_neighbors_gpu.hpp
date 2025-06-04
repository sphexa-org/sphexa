#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/traversal/groups.hpp"

#include "sph/particles_data.hpp"

namespace sph
{

void findNeighborsSfc(const cstone::GroupView& groups, sphexa::ParticlesData<cstone::GpuTag>& d,
                      const cstone::Box<SphTypes::CoordinateType>& box, bool subgroups = false);

}
