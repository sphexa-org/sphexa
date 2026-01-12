#pragma once

#include <cmath>

#include "cstone/cuda/cuda_utils.hpp"
#include "sph/kernels.hpp"
#include "sph/sph_gpu.hpp"

namespace sph
{

template<class T, class KeyType>
bool updateSmoothingLengthCpu(size_t startIndex, size_t endIndex, unsigned ng0, const unsigned* nc, T* h, KeyType* keys)
{
    bool keysRemoved = false;
#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        if (h[i] == 0)
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
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        bool keysRemoved =
            updateSmoothingLengthGpu(grp, d.ng0, rawPtr(d.nc), rawPtr(d.h), rawPtr(d.keys));
        syncGpu();
        return keysRemoved;
    }
    else
    {
        return updateSmoothingLengthCpu(grp.firstBody, grp.lastBody, d.ng0, rawPtr(d.nc), rawPtr(d.h), rawPtr(d.keys));
    }
}

} // namespace sph
