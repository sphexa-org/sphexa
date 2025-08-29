//
// Created by Noah Kubli on 04.03.2025.
//

#pragma once

#include <cmath>
#include "cstone/primitives/stl.hpp"
#include "cstone/tree/definitions.h"

namespace disk
{

enum StarPotentialType : int
{
    newtonian = 0,
    //! @brief Post-Newtonian potential used in Nelson & Papaloizou 2000
    //! Hydrodynamic simulations of the Bardeen-Petterson effect
    einstein_precession = 1
};

template<typename Treal, typename Thydro, typename Tmass>
struct CentralForceData
{
    const Treal *        x, *y, *z;
    const Tmass*         m;
    Thydro *             ax, *ay, *az;
    Treal                g;
    const double         m_star;
    const double         inner_size2;
    const double         c_light;
    cstone::Vec3<double> star_position;
};

template<typename Data>
HOST_DEVICE_FUN void newtonianGravity(const Data& d, size_t i, cstone::Vec4<double>& star_force_local, float& t_star)
{
    const double dx    = d.x[i] - d.star_position[0];
    const double dy    = d.y[i] - d.star_position[1];
    const double dz    = d.z[i] - d.star_position[2];
    const double dist2 = stl::max(d.inner_size2, dx * dx + dy * dy + dz * dz);
    const double dist  = std::sqrt(dist2);
    const double dist3 = dist2 * dist;

    const double a_strength = 1. / dist3 * d.m_star * d.g;
    const double ax_i       = -dx * a_strength;
    const double ay_i       = -dy * a_strength;
    const double az_i       = -dz * a_strength;
    d.ax[i] += ax_i;
    d.ay[i] += ay_i;
    d.az[i] += az_i;

    star_force_local[0] -= d.g * d.m[i] / dist;
    star_force_local[1] -= ax_i * d.m[i];
    star_force_local[2] -= ay_i * d.m[i];
    star_force_local[3] -= az_i * d.m[i];
    t_star = stl::min(t_star, float(std::sqrt(dist / a_strength)));
}

template<typename Data>
HOST_DEVICE_FUN void einsteinPrecession(const Data& d, size_t i, cstone::Vec4<double>& star_force_local, float& t_star)
{
    const double dx    = d.x[i] - d.star_position[0];
    const double dy    = d.y[i] - d.star_position[1];
    const double dz    = d.z[i] - d.star_position[2];
    const double dist2 = stl::max(d.inner_size2, dx * dx + dy * dy + dz * dz);
    const double dist  = std::sqrt(dist2);
    const double dist3 = dist2 * dist;

    const double r_grav            = d.g * d.m_star / (d.c_light * d.c_light);
    const double correction_factor = 1. + 6. * r_grav / dist;

    const double a_strength = 1. / dist3 * d.m_star * d.g * correction_factor;
    const double ax_i       = -dx * a_strength;
    const double ay_i       = -dy * a_strength;
    const double az_i       = -dz * a_strength;
    d.ax[i] += ax_i;
    d.ay[i] += ay_i;
    d.az[i] += az_i;

    star_force_local[0] -= d.g * d.m[i] / dist;
    star_force_local[1] -= ax_i * d.m[i];
    star_force_local[2] -= ay_i * d.m[i];
    star_force_local[3] -= az_i * d.m[i];
    t_star = stl::min(t_star, float(std::sqrt(dist / a_strength)));
}
} // namespace disk
