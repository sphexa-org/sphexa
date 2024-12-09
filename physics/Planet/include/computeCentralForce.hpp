//
// Created by Noah Kubli on 04.03.2024.
//

#pragma once

#include <algorithm>
#include <cmath>

#include "cstone/tree/accel_switch.hpp"
#include "cstone/tree/definitions.h"
#include "computeCentralForce_gpu.hpp"

namespace disk
{

template<typename Dataset, typename StarData>
void computeCentralForceImpl(size_t first, size_t last, Dataset& d, StarData& star)
{
    cstone::Vec4<double> force_local{};
    const double         inner_size2 = star.inner_size * star.inner_size;

#pragma omp declare reduction(add_force : cstone::Vec4 <double> : omp_out = omp_out + omp_in) initializer(omp_priv = {})

#pragma omp parallel for reduction(add_force : force_local)
    for (size_t i = first; i < last; i++)
    {
        const double dx    = d.x[i] - star.position[0];
        const double dy    = d.y[i] - star.position[1];
        const double dz    = d.z[i] - star.position[2];
        const double dist2 = std::max(inner_size2, dx * dx + dy * dy + dz * dz);
        const double dist  = std::sqrt(dist2);
        const double dist3 = dist2 * dist;

        const double a_strength = 1. / dist3 * star.m * d.g;
        const double ax_i       = -dx * a_strength;
        const double ay_i       = -dy * a_strength;
        const double az_i       = -dz * a_strength;
        d.ax[i] += ax_i;
        d.ay[i] += ay_i;
        d.az[i] += az_i;

        force_local[0] -= d.g * d.m[i] / dist;
        force_local[1] -= ax_i * d.m[i];
        force_local[2] -= ay_i * d.m[i];
        force_local[3] -= az_i * d.m[i];
    }

    star.force_local = force_local;
}

template<typename Dataset, typename StarData>
void computeCentralForce(size_t startIndex, size_t endIndex, Dataset& d, StarData& star)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        computeCentralForceGPU(startIndex, endIndex, d, star);
    }
    else { computeCentralForceImpl(startIndex, endIndex, d, star); }
}

} // namespace disk
