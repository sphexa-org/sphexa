//
// Created by Noah Kubli on 17.04.2024.
//

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include "betaCooling_gpu.hpp"
#include "cstone/primitives/accel_switch.hpp"
#include "get_ptr.hpp"

namespace disk
{

template<typename Dataset, typename StarData>
void betaCoolingImpl(size_t first, size_t last, Dataset& d, const StarData& star)
{
#pragma omp parallel for
    for (size_t i = first; i < last; i++)
    {
        if (d.rho[i] < star.cooling_rho_limit && d.u[i] > star.u_floor)
        {
            const double dx    = d.x[i] - star.position[0];
            const double dy    = d.y[i] - star.position[1];
            const double dz    = d.z[i] - star.position[2];
            const double dist2 = dx * dx + dy * dy + dz * dz;
            const double dist  = std::sqrt(dist2);
            const double omega = std::sqrt(d.g * star.m / (dist2 * dist));
            d.du[i] += -d.u[i] * omega / star.beta;
        }
    }
}

template<typename Dataset>
auto duTimestepImpl(size_t first, size_t last, const Dataset& d)
{

    using Tu         = std::decay_t<decltype(d.u[0])>;
    using Tdu        = std::decay_t<decltype(d.du[0])>;
    using Tt         = std::common_type_t<Tu, Tdu>;
    Tt duTimestepMin = std::numeric_limits<Tt>::infinity();

#pragma omp parallel for reduction(min : duTimestepMin)
    for (size_t i = first; i < last; i++)
    {
        Tt duTimestep = std::abs(d.u[i] / d.du[i]);
        duTimestepMin = std::min(duTimestepMin, duTimestep);
    }
    return duTimestepMin;
}

//! @brief Cool the disk with a cooling time proportional to the Keplerian orbital time (around the star)
template<typename Dataset, typename StarData>
void betaCooling(size_t startIndex, size_t endIndex, Dataset& d, const StarData& star)
{
    using T_beta = std::decay_t<decltype(star.beta)>;
    if (star.beta != std::numeric_limits<T_beta>::infinity())
    {
        if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
        {
            betaCoolingGPU(startIndex, endIndex, getPtr<"x">(d), getPtr<"y">(d), getPtr<"z">(d), getPtr<"u">(d),
                           getPtr<"rho">(d), getPtr<"du">(d), d.g, star);
        }
        else { betaCoolingImpl(startIndex, endIndex, d, star); }
    }
}

//! @brief Compute a maximal time step depending on how fast the internal energy changes
//! (e.g. due to cooling).
template<typename Dataset, typename StarData>
void duTimestep(size_t startIndex, size_t endIndex, Dataset& d, StarData& star)
{
    if (star.K_u == std::numeric_limits<decltype(star.K_u)>::infinity())
    {
        star.t_du = std::numeric_limits<decltype(star.t_du)>::infinity();
    }
    else
    {
        if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
        {
            star.t_du = star.K_u * duTimestepGPU(startIndex, endIndex, getPtr<"u">(d), getPtr<"du">(d));
        }
        else { star.t_du = star.K_u * duTimestepImpl(startIndex, endIndex, d); }
    }
}

} // namespace disk
