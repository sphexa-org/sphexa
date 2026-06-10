#pragma once

#include <cmath>
#include <iostream>
#include <vector>

#include "cstone/cuda/cuda_utils.hpp"
#include "sph/kernels.hpp"
#include "sph/sph_gpu.hpp"

namespace sph
{

using cstone::LocalIndex;

template<class T, class KeyType>
bool updateSmoothingLengthCpu(size_t startIndex, size_t endIndex, unsigned ng0, const unsigned* nc, T* h, KeyType* keys)
{
    bool keysRemoved = false;
#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        if (nc[i] <= 1)
        {
            keys[i]     = cstone::removeKey<KeyType>{};
            keysRemoved = true;
        }
        h[i] = updateH(ng0, nc[i], h[i]);

#ifndef NDEBUG
        if (std::isinf(h[i]) || std::isnan(h[i])) printf("ERROR::h(%lu) ngi %d h %f\n", i, nc[i], h[i]);
#endif
    }
    return keysRemoved;
}

template<class Dataset>
bool updateSmoothingLength(const GroupView& grp, Dataset& d)
{
    using namespace cstone;
    if constexpr (execution::HaveGpu<typename Dataset::Exec>{})
    {
        bool keysRemoved = updateSmoothingLengthGpu(grp, d.ng0, rawPtr(d.nc), rawPtr(d.h), rawPtr(d.keys));
        syncGpu(0);
        return keysRemoved;
    }
    else
    {
        return updateSmoothingLengthCpu(grp.firstBody, grp.lastBody, d.ng0, rawPtr(d.nc), rawPtr(d.h), rawPtr(d.keys));
    }
}

template<class Tc, class T, class KeyType>
void updateSmoothingLengthIterativeCpu(const Tc* x, const Tc* y, const Tc* z, T* h, unsigned* nc, LocalIndex firstId,
                                       LocalIndex lastId, const cstone::Box<Tc>& box,
                                       const cstone::OctreeNsView<Tc, KeyType>& treeView, unsigned ng0, unsigned ngmax)
{
    LocalIndex numWork = lastId - firstId;

    unsigned ngmin = ng0 / 4;

    constexpr int maxIteration = 10;

#pragma omp parallel
    {
        std::vector<LocalIndex> neighbors(ngmax);

#pragma omp for
        for (LocalIndex i = 0; i < numWork; ++i)
        {
            LocalIndex id    = i + firstId;
            unsigned   ncSph = 1 + findNeighbors(id, x, y, z, h, treeView, box, ngmax, neighbors.data());

            int iteration = 0;
            while ((ngmin > ncSph || (ncSph - 1) > ngmax) && iteration++ < maxIteration)
            {
                h[id] = updateH(ng0, ncSph, h[id]);
                ncSph = 1 + findNeighbors(id, x, y, z, h, treeView, box, ngmax, neighbors.data());
            }

            nc[id] = ncSph;

            if (iteration == maxIteration && (ngmin > ncSph || (ncSph - 1) > ngmax)) { nc[id] = 1; }
        }
    }
}

template<class T, class Dataset>
void updateSmoothingLengthIterative(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (cstone::execution::HaveGpu<typename Dataset::Exec>{})
    {
        updateSmoothingLengthIterativeGpu(groups, d, box);
    }
    else
    {
        updateSmoothingLengthIterativeCpu(d.x.data(), d.y.data(), d.z.data(), d.h.data(), d.nc.data(), groups.firstBody,
                                          groups.lastBody, box, d.treeView, d.ng0, d.ngmax);
    }
}

} // namespace sph
