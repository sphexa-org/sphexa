#pragma once

#include "cstone/findneighbors.hpp"
#include "cstone/traversal/groups.hpp"
#include "cstone/traversal/ijloop/cpu.hpp"

#include "sph/find_neighbors_gpu.hpp"
#include "sph/neighborhood.hpp"

namespace sph
{

//! @brief perform neighbor search together with updating the smoothing lengths
template<class T, class Dataset>
void findNeighborsSfc(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box, bool symmetric = false,
                      bool clustered = true)
{
    if (d.ng0 > d.ngmax) { throw std::runtime_error("ng0 should be smaller than ngmax\n"); }

    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        findNeighborsSfcGpu(groups, d, box, symmetric, clustered);
    }
    else { buildNeighborhood(groups, d, box); }
}

} // namespace sph
