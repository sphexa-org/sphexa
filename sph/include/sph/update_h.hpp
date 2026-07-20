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
