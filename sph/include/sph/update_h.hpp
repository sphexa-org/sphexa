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

template<class Tc, class T, class KeyType>
bool updateSmoothingLengthIterativeCpu(const Tc* x, const Tc* y, const Tc* z, T* h, unsigned* nc, KeyType* keys,
                                       LocalIndex firstId, LocalIndex lastId, const cstone::Box<Tc>& box,
                                       const cstone::OctreeNsView<Tc, KeyType>& treeView, unsigned ng0, unsigned ngmax)
{
    bool keysRemoved = false;
#pragma omp parallel for
    for (LocalIndex i = firstId; i < lastId; ++i)
    {
        h[i] = updateH(ng0, nc[i], h[i]);
        updateHIterative(ng0, ngmax, box, treeView, i, x, y, z, h, nc);
        if (nc[i] <= 1)
        {
            keys[i]     = cstone::removeKey<KeyType>{};
            keysRemoved = true;
        }
    }
    return keysRemoved;
}

template<class T, class Dataset>
bool updateSmoothingLengthIterative(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (d.useGpu) { return updateSmoothingLengthIterativeGpu(groups, d, box); }
    else
    {
        return updateSmoothingLengthIterativeCpu(d.x.data(), d.y.data(), d.z.data(), d.h.data(), d.nc.data(),
                                                 d.keys.data(), groups.firstBody, groups.lastBody, box, d.treeView,
                                                 d.ng0, d.ngmax);
    }
}

} // namespace sph
