#pragma once

#include "cstone/findneighbors.hpp"
#include "cstone/traversal/groups.hpp"
#include "cstone/traversal/ijloop/cpu.hpp"

#include "sph/find_neighbors_gpu.hpp"

namespace sph
{

using cstone::LocalIndex;

template<class Tc, class T, class KeyType>
void findNeighborsSph(const Tc* x, const Tc* y, const Tc* z, T* h, LocalIndex firstId, LocalIndex lastId,
                      const cstone::Box<Tc>& box, const cstone::OctreeNsView<Tc, KeyType>& treeView, unsigned ngmax,
                      LocalIndex* neighbors)
{
    LocalIndex numWork = lastId - firstId;

#pragma omp parallel for
    for (LocalIndex i = 0; i < numWork; ++i)
    {
        LocalIndex id = i + firstId;
        findNeighbors(id, x, y, z, h, treeView, box, ngmax, neighbors + i * ngmax);
    }
}

//! @brief perform neighbor search together with updating the smoothing lengths
template<class T, class Dataset>
void findNeighborsSfc(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box)
{
    if (d.ng0 > d.ngmax) { throw std::runtime_error("ng0 should be smaller than ngmax\n"); }

    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{}) { findNeighborsSfcGpu(groups, d, box); }
    else
    {
        findNeighborsSph(d.x.data(), d.y.data(), d.z.data(), d.h.data(), groups.firstBody, groups.lastBody, box,
                         d.treeView, d.ngmax, d.neighbors.data());
        d.neighborhood = cstone::ijloop::CpuFullNbListNeighborhood{d.ngmax}.build(
            d.treeView, box, d.size(), groups, d.x.data(), d.y.data(), d.z.data(), d.h.data());
    }
}

} // namespace sph
