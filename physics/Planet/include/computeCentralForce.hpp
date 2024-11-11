//
// Created by Noah Kubli on 04.03.2024.
//

#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <mpi.h>
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/fields/field_get.hpp"
#include "cstone/tree/accel_switch.hpp"
#include "cstone/cuda/cuda_stubs.h"

#include "computeCentralForce_gpu.hpp"
#include "accretion_gpu.hpp"
#include "accretion_impl.hpp"
#include "sph/particles_data.hpp"

namespace planet
{

template<typename Dataset, typename StarData>
void computeCentralForceImpl(size_t first, size_t last, Dataset& d, StarData& star)
{
    using Tf    = typename decltype(star.force_local)::value_type;
    Tf force[3] = {};

    using Tp = std::decay_t<decltype(star.potential_local)>;
    Tp potential{0.};

    const double inner_size2 = star.inner_size * star.inner_size;

#pragma omp parallel for reduction(+ : force[ : 3]) reduction(+ : potential)
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

        force[0] -= ax_i * d.m[i];
        force[1] -= ay_i * d.m[i];
        force[2] -= az_i * d.m[i];
        potential -= d.g * d.m[i] / dist;
    }

    star.force_local[0] = force[0];
    star.force_local[1] = force[1];
    star.force_local[2] = force[2];
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

} // namespace planet
