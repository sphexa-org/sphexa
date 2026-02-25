#pragma once

#include <cmath>

#include "cstone/cuda/cuda_utils.hpp"
#include "sph/kernels.hpp"
#include "sph/sph_gpu.hpp"

namespace sph
{

template<class T, class KeyType>
unsigned long long updateSmoothingLengthCpu(size_t startIndex, size_t endIndex, unsigned ng0, const unsigned* nc, T* h, KeyType* keys)
{
    unsigned long long n_removed = 0;
#pragma omp parallel for schedule(static) reduction(+: n_removed)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        if (nc[i] <= 1)
        {
            keys[i]     = cstone::removeKey<KeyType>{};
            n_removed++;
        }
        h[i] = updateH(ng0, nc[i], h[i]);

#ifndef NDEBUG
        if (std::isinf(h[i]) || std::isnan(h[i])) printf("ERROR::h(%lu) ngi %d h %f\n", i, nc[i], h[i]);
#endif
    }
    return n_removed;
}

template<class Dataset>
unsigned long long updateSmoothingLength(const GroupView& grp, Dataset& d)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        unsigned long long keysRemoved =
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
