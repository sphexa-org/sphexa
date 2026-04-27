//
// Created by Noah Kubli on 06.03.2025.
//

#pragma once

#include <tuple>
#include "interpolator.hpp"

namespace polytrope
{

/*! @brief Compute the profile of a polytrope in hydrostatic equilibrium. Returns rho(r), r(M_enclosed) and the polytropic constant K.
 * @param polytropic_n polytropic exponent n = 1 / (gamma - 1)
 * @param total_mass Mass of the polytrope
 * @param radial_size radial size of the polytrope
 * @param G gravitational constant
 */
std::tuple<LinearInterpolator, LinearInterpolator, double>
computePolytropeProfile(double polytropic_n, double total_mass, double radial_size, double G);

} // namespace polytrope
