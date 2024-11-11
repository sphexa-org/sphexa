/*! @file
 * @brief Polytropic equation of state
 *
 * @author Noah Kubli <noah.kubli@uzh.ch>
 */

#pragma once

#include "cstone/cuda/cuda_utils.hpp"
#include "eos_polytropic_gpu.hpp"
#include "eos_polytropic_loop.hpp"
#include "sph/particles_data_stubs.hpp"
#include "sph/eos.hpp"

namespace planet
{

template<typename Dataset, typename StarData>
void computePolytropic_HydroStdImpl(size_t startIndex, size_t endIndex, Dataset& d, const StarData& star)
{
    const auto* rho = d.rho.data();

    auto* p = d.p.data();
    auto* c = d.c.data();

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        std::tie(p[i], c[i]) = polytropicEOS(star.Kpoly, star.exp_poly, d.gamma, rho[i]);
    }
}

template<class Dataset, typename StarData>
void computePolytropicEOS_HydroStd(size_t startIndex, size_t endIndex, Dataset& d, const StarData& star)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        computePolytropicEOS_HydroStdGPU(startIndex, endIndex, d, star);
    }
    else { computePolytropic_HydroStdImpl(startIndex, endIndex, d, star); }
}

} // namespace planet
