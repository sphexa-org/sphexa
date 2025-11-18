//
// Created by Noah Kubli on 06.03.2025.
//

#pragma once

#include <tuple>
#include <utility>
#include <vector>

#include "cstone/util/array.hpp"
#include "interpolator.hpp"
#include "integrate.hpp"
#include "lane_emden.hpp"

namespace polytrope
{

auto transformToPhysicalValues(const std::vector<double>& xi, const std::vector<util::array<double, 2>>& theta_phi,
                               double polytropic_n, double rho_c, double K, double G)
{
    std::vector<double> density(theta_phi.size());
    auto                theta_phi_to_rho = [rho_c, polytropic_n](const auto& theta_phi)
    { return rho_c * std::pow(theta_phi[0], polytropic_n); };
    std::transform(theta_phi.begin(), theta_phi.end(), density.begin(), theta_phi_to_rho);

    std::vector<double> enclosed_mass(theta_phi.size());
    auto                value_to_encl_m = [rho_c, K, polytropic_n, G](const auto& theta_phi)
    {
        const double phi = theta_phi[1]; // phi = -xi^2 * dtheta / dxi
        return get_enclosed_mass(polytropic_n, phi, K, rho_c, G);
    };
    std::transform(theta_phi.begin(), theta_phi.end(), enclosed_mass.begin(), value_to_encl_m);

    std::vector<double> radius(theta_phi.size());
    const auto          alpha_c = alpha(polytropic_n, rho_c, K, G);
    auto                xi_to_r = [alpha_c](const auto& xi) { return alpha_c * xi; };
    std::transform(xi.begin(), xi.end(), radius.begin(), xi_to_r);

    return std::tuple{std::move(radius), std::move(density), std::move(enclosed_mass)};
}

/*! @brief Compute the profile of a polytrope. Returns rho(r), r(M_enclosed) and the polytropic constant K.
 * @param polytropic_n polytropic exponent n = 1 / (gamma - 1)
 * @param total_mass Mass of the polytrope
 * @param radial_size radial size of the polytrope
 * @param G gravitational constant
 */
auto computePolytropeProfile(double polytropic_n, double total_mass, double radial_size, double G)
{
    // Start with an asymptotic expansion of the solution from 0.
    LaneEmdenAsymptoticStart asympt{polytropic_n};
    constexpr double         xi_asympt_limit = 1e-8;
    constexpr double         xi_max          = 20.0;
    constexpr double         min_step_size   = 1e-2;
    constexpr double         max_error       = 1e-12;

    const auto [xi, theta_phi] = integrate_to_zero(LaneEmden{polytropic_n}, asympt(1., xi_asympt_limit),
                                                   xi_asympt_limit, xi_max, min_step_size, max_error);

    //! @brief the first root of theta
    const double xi_1        = xi.back();
    const double dtheta_xi_1 = -theta_phi.back()[1] / (xi.back() * xi.back());

    const double rho_c = get_rho_c(xi_1, dtheta_xi_1, radial_size, total_mass);
    const double K     = get_K(polytropic_n, xi_1, dtheta_xi_1, radial_size, rho_c, G);

    auto [radius, density, enclosed_mass] = transformToPhysicalValues(xi, theta_phi, polytropic_n, rho_c, K, G);

    return std::tuple{LinearInterpolator{radius, density}, LinearInterpolator{enclosed_mass, radius}, K};
}
} // namespace polytrope