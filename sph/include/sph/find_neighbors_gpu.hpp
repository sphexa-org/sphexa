#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/traversal/groups.hpp"

#include "sph/particles_data.hpp"

namespace sph
{

void findNeighborsSfc(const cstone::GroupView& groups, sphexa::ParticlesData<cstone::execution::Gpu>& d,
                      const cstone::Box<SphTypes::CoordinateType>& box, bool subgroups = false);

template<class Th, class KeyType>
extern bool updateSmoothingLengthGpu(const cstone::GroupView&, unsigned ng0, const unsigned* nc, Th* h, KeyType* keys);

template<class T, class Dataset>
extern void updateSmoothingLengthIterativeGpu(const cstone::GroupView&, Dataset& d, const cstone::Box<T>& box);

}
