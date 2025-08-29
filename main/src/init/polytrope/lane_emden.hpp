//
//  lane_emden.hpp
//  Polytrop
//
//  Created by Noah Kubli on 14.02.2025.
//

#pragma once

#include "cstone/util/array.hpp"
#include <cmath>

namespace polytrope
{
//! @brief System of differential equations representing the Lane-Emden equation.
//! transformed into theta(xi) and phi(xi)
//! dtheta/dxi = -phi / xi^2; dphi/dxi = theta^n * xi^2
//! y = [theta, phi]
struct LaneEmden
{
    const double polytropic_n = 1.5;
    auto         operator()(util::array<double, 2> y, double t) const
    {
        const double n = polytropic_n;
        if (std::floor(n) == n)
            return util::array<double, 2>{-y[1] / (t * t), t * t * std::pow(y[0], n)};
        else
            return util::array<double, 2>{-y[1] / (t * t), t * t * std::pow(std::abs(y[0]), n)};
    }
};

//! @brief Asymptotic solution of the Lane-Emden equation for xi ~ 0
struct LaneEmdenAsymptoticStart
{
    const double polytropic_n = 1.5;
    auto         operator()(double theta_0, double t) const
    {
        const double n = polytropic_n;
        return util::array<double, 2>{theta_0 - std::pow(theta_0, n) / 6. * t * t,
                                      -std::pow(theta_0, n) / 3. * t * t * t};
    }
};

double get_enclosed_mass(double polytropic_n, double phi, double K, double rho_c, double G)
{
    const double n       = polytropic_n;
    const double pre_fac = K / G * (n + 1.) / (4. * M_PI);
    const double rho_fac = std::pow(rho_c, (3. - n) / (2. * n));
    return 4. * M_PI * std::pow(pre_fac, 1.5) * rho_fac * phi;
};

//! @brief A characteristic length
double alpha(double polytropic_n, double rho_c, double K, double G)
{
    const double n       = polytropic_n;
    const double rho_fac = std::pow(rho_c, (1. - n) / n);
    return std::sqrt(K * (n + 1.) * rho_fac / (4. * M_PI * G));
};

//! @brief The central density
//! @param xi_1: Value of xi at first root of theta
//! @param dtheta: dtheta/dxi at xi_1
double get_rho_c(double xi_1, double dtheta, double star_radius, double star_mass)
{
    const double density = star_mass / (4. * M_PI * star_radius * star_radius * star_radius);
    return density / (-dtheta / xi_1);
};

//! @brief The polytropic constant
double get_K(double polytropic_n, double xi_1, double dtheta, double star_radius, double rho_c, double G)
{
    const double n        = polytropic_n;
    const double rho_term = std::pow(rho_c, (n - 1.) / n);
    return star_radius * star_radius * G * 4. * M_PI / (n + 1.) * rho_term / (xi_1 * xi_1);
};
} // namespace polytrope