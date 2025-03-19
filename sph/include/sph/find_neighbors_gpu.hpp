#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/traversal/groups.hpp"

namespace sph
{

template<class T, class Dataset>
extern void findNeighborsSfcGpu(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box, bool symmetric);

}
