#pragma once

#include "cstone/findneighbors.hpp"

namespace sph
{

using cstone::LocalIndex;

template<class Tc, class T, class KeyType>
void findNeighborsSph(const Tc* x, const Tc* y, const Tc* z, T* h, LocalIndex firstId, LocalIndex lastId,
                      const cstone::Box<Tc>& box, const cstone::OctreeNsView<Tc, KeyType>& treeView, unsigned ng0,
                      unsigned ngmax, LocalIndex* neighbors, unsigned* nc)
{
    LocalIndex numWork = lastId - firstId;

    unsigned ngmin = ng0 / 4;

    constexpr int maxIteration = 10;

#pragma omp parallel for
    for (LocalIndex i = 0; i < numWork; ++i)
    {
        LocalIndex id    = i + firstId;
        unsigned   ncSph = 1 + findNeighbors(id, x, y, z, h, treeView, box, ngmax, neighbors + i * ngmax);

        int iteration = 0;
        while ((ngmin > ncSph || (ncSph - 1) > ngmax) && iteration++ < maxIteration)
        {
            h[id] = updateH(ng0, ncSph, h[id]);
            ncSph = 1 + findNeighbors(id, x, y, z, h, treeView, box, ngmax, neighbors + i * ngmax);
        }
        nc[i] = ncSph;

        if (iteration == maxIteration && (ngmin > ncSph || (ncSph - 1) > ngmax))
        {
            h[i]  = T(0);
            nc[i] = 1;
        }
    }
}

//! @brief perform neighbor search together with updating the smoothing lengths
template<class T, class Dataset>
void findNeighborsSfc(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{}) { return; }

    if (d.ng0 > d.ngmax) { throw std::runtime_error("ng0 should be smaller than ngmax\n"); }

    findNeighborsSph(d.x.data(), d.y.data(), d.z.data(), d.h.data(), startIndex, endIndex, box, d.treeView, d.ng0,
                     d.ngmax, d.neighbors.data(), d.nc.data() + startIndex);
}

} // namespace sph
