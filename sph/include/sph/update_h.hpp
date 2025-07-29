#pragma once

#include <cmath>

#include "cstone/cuda/cuda_utils.hpp"
#include "sph/kernels.hpp"
#include "sph/sph_gpu.hpp"

namespace sph
{

template<class T>
void updateSmoothingLengthCpu(size_t startIndex, size_t endIndex, unsigned ng0, const unsigned* nc, T* h)
{
#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        h[i] = updateH(ng0, nc[i], h[i]);

#ifndef NDEBUG
        if (std::isinf(h[i]) || std::isnan(h[i])) printf("ERROR::h(%lu) ngi %d h %f\n", i, nc[i], h[i]);
#endif
    }
}

template<class Dataset>
void updateSmoothingLength(const GroupView& grp, Dataset& d)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        updateSmoothingLengthGpu(grp, d.ng0, rawPtr(d.devData.nc), rawPtr(d.devData.h));
        syncGpu();
    }
    else { updateSmoothingLengthCpu(grp.firstBody, grp.lastBody, d.ng0, rawPtr(d.nc), rawPtr(d.h)); }
}

} // namespace sph
