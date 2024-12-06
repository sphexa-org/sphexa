#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/traversal/groups.hpp"
#include "cstone/tree/octree.hpp"
#include "cstone/tree/definitions.h"
#include "sph/timestep.h"

namespace disk
{
template<typename Dataset, typename StarData>
extern void computePolytropicEOS_HydroStdGPU(size_t firstParticle, size_t lastParticle, Dataset& d,
                                             const StarData& star);
} // namespace disk
