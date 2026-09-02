#pragma once

#include "cstone/traversal/groups.hpp"

#include "sph/find_neighbors_gpu.hpp"
#include "sph/neighborhood.hpp"
#include "sph/particles_data.hpp"

namespace sph
{

inline void findNeighborsSfc(const cstone::GroupView& groups, sphexa::ParticlesData<cstone::execution::Cpu>& d,
                             const cstone::Box<SphTypes::CoordinateType>& box, bool subgroups = false)
{
    if (d.ng0 > d.ngmax) { throw std::runtime_error("ng0 should be smaller than ngmax\n"); }

    d.neighborhood.build(groups, d, box, subgroups);
}

using cstone::GroupView;
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
    if constexpr (d.useGpu)
    {
        bool keysRemoved = updateSmoothingLengthGpu(grp, d.ng0, rawPtr(d.nc), rawPtr(d.h), rawPtr(d.keys));
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
#pragma omp parallel for
    for (LocalIndex i = firstId; i < lastId; ++i)
    {
        updateHIterative(ng0, ngmax, box, treeView, i, x, y, z, h, nc);
    }
}

template<class T, class Dataset>
void updateSmoothingLengthIterative(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (d.useGpu) { updateSmoothingLengthIterativeGpu(groups, d, box); }
    else
    {
        updateSmoothingLengthIterativeCpu(d.x.data(), d.y.data(), d.z.data(), d.h.data(), d.nc.data(), groups.firstBody,
                                          groups.lastBody, box, d.treeView, d.ng0, d.ngmax);
    }
}

} // namespace sph
