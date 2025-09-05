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

    size_t        numFails     = 0;
    constexpr int maxIteration = 10;

#pragma omp parallel for reduction(+ : numFails)
    for (LocalIndex i = 0; i < numWork; ++i)
    {
        LocalIndex id    = i + firstId;
        unsigned   ncSph = 1 + findNeighbors(id, x, y, z, h, treeView, box, ngmax, neighbors + i * ngmax);
        // std::cout << "[findNeighborsSph] particle " << id << " initial ncSph: " << ncSph << " h: " << h[id] << std::endl;
        // std::cout << "[findNeighborsSph] ngmin: " << ngmin << " ngmax: " << ngmax << " ng0: " << ng0 << std::endl;

        int iteration = 0;
        while ((ngmin > ncSph || (ncSph - 1) > ngmax) && iteration++ < maxIteration)
        {
            h[id] = updateH(ng0, ncSph, h[id]);
            ncSph = 1 + findNeighbors(id, x, y, z, h, treeView, box, ngmax, neighbors + i * ngmax);
            // std::cout << "[findNeighborsSph] particle " << id << " iteration: " << iteration << " ncSph: " << ncSph
            //           << " h: " << h[id] << std::endl;
        }
        // if (iteration >= maxIteration) {
        //     std::cout << "Warning: findNeighborsSph did not converge for particle " << id << " ncSph: " << ncSph
        //               << " h: " << h[id] << std::endl;
        // }
        numFails += (iteration >= maxIteration);

        nc[i] = ncSph;
        // std::cout << "[findNeighborsSph] particle " << id << " final ncSph: " << ncSph << " h: " << h[id] << std::endl;
    }

    if (numFails)
    {
        std::cout << "Coupled h-neighbor count updated failed to converge for " << numFails << " particles"
                  << std::endl;
    }
}

//! @brief perform neighbor search together with updating the smoothing lengths
template<class T, class Dataset>
void findNeighborsSfc(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{}) { return; }

    if (d.ng0 > d.ngmax) { throw std::runtime_error("ng0 should be smaller than ngmax\n"); }

    // std::cout << "[findNeighborsSfc] Finding neighbors for particles in index range [" << startIndex << ", " << endIndex
    //           << ")" << std::endl;

    // std::cout << "[findNeighborsSfc] box: (" << box.xmin() << ", " << box.xmax() << ") x (" << box.ymin() << ", " << box.ymax()
    //           << ") x (" << box.zmin() << ", " << box.zmax() << ")" << std::endl;
    
    for (size_t i = 0; i < d.treeView.numLeafNodes; ++i)
    {
        const auto internal_index = i;
        if (d.treeView.sizes[internal_index][0] == 0 && d.treeView.sizes[internal_index][1] == 0 &&
            d.treeView.sizes[internal_index][2] == 0)
        {
            continue;
        }
        // std::cout << "[findNeighborsSfc] prefix[" << internal_index << "]: " << std::oct << d.treeView.prefixes[internal_index] << std::dec
        //           << " centers[" << internal_index << "]: (" << d.treeView.centers[internal_index][0] << ", "
        //           << d.treeView.centers[internal_index][1] << ", " << d.treeView.centers[internal_index][2] << ") sizes[" << internal_index
        //           << "]: (" << d.treeView.sizes[internal_index][0] << ", " << d.treeView.sizes[internal_index][1] << ", "
        //           << d.treeView.sizes[internal_index][2] << ")" << std::endl;
    }

    findNeighborsSph(d.x.data(), d.y.data(), d.z.data(), d.h.data(), startIndex, endIndex, box, d.treeView, d.ng0,
                     d.ngmax, d.neighbors.data(), d.nc.data() + startIndex);
}

} // namespace sph
