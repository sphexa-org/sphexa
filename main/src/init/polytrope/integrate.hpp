//
//  integrate.hpp.h
//  Polytrop
//
//  Created by Noah Kubli on 14.02.2025.
//

#pragma once

#include "cstone/util/array.hpp"

#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

namespace polytrope
{
//! @brief One step of Runge-Kutta 4 integration
template<typename F, size_t N>
auto rk4(F&& func, const util::array<double, N>& y, double t, double delta_t)
{
    const auto k1 = func(y, t);
    const auto k2 = func(y + k1 * (delta_t / 2.), t + delta_t / 2.);
    const auto k3 = func(y + k2 * (delta_t / 2.), t + delta_t / 2.);
    const auto k4 = func(y + k3 * delta_t, t + delta_t);

    return y + (k1 + k2 * 2. + k3 * 2. + k4) * (1. / 6.) * delta_t;
}

//! @brief Integrate a step and estimate the error
template<typename F, size_t N>
auto integrate_step(F&& func, double t, double delta_t, const util::array<double, N> y)
{
    const double t_new            = t + delta_t;
    const auto   y_new_0          = rk4(func, y, t, delta_t);
    const auto   y_new_1_halfstep = rk4(func, y, t, delta_t / 2.);
    const auto   y_new_1          = rk4(func, y_new_1_halfstep, t + delta_t / 2., delta_t / 2.);

    const double error = max(abs(abs(y_new_1) - abs(y_new_0)));
    return std::make_tuple(t_new, y_new_1, error);
}

double new_step_size_factor(double error, double target_error)
{
    if (std::isnan(error)) { return 0.5; }
    return std::max(0.5, std::min(1.5, std::sqrt(target_error / (error))));
}

template<size_t N>
bool has_nan(const util::array<double, N>& a) noexcept
{
    for (size_t i = 0; i < N; i++)
    {
        if (std::isnan(a[i])) { return true; }
    }
    return false;
}

//! @brief Integrate func until the first zero using an adaptive timestep.
//! Returns times, values
template<typename F, size_t N>
auto integrate_to_zero(F&& func, const util::array<double, N>& y_0, double t_0, double t_end,
                       double min_step_size = 1e-3, double maximal_error = 0.01)
{
    const double target_error = 0.5 * maximal_error;

    double delta_t = 0.01;

    auto get_estimated_steps = [](double integration_distance, double delta_t)
    { return integration_distance / delta_t; };
    const double n_estimated = get_estimated_steps(t_end - t_0, delta_t);

    std::vector<double>                 t{t_0};
    std::vector<util::array<double, N>> y{y_0};
    t.reserve(n_estimated);
    y.reserve(n_estimated);

    auto close_to_zero = [&](double x) { return std::abs(x) < maximal_error; };

    size_t i        = 0;
    size_t it_count = 0;
    while (t.back() < t_end && !close_to_zero(y.back()[0]))
    {
        delta_t                          = std::min(min_step_size, std::min(delta_t, t_end - t.back()));
        const auto [t_new, y_new, error] = integrate_step(func, t[i], delta_t, y[i]);

        double new_delta_t = delta_t * new_step_size_factor(error, target_error);
        if (has_nan(y_new) || y_new[0] < 0.) { new_delta_t = delta_t / 2.; }

        bool accept = error <= maximal_error && !has_nan(y_new) && !std::isnan(error) && y_new[0] >= 0.;

        if (accept)
        {
            t.push_back(t_new);
            y.push_back(y_new);
            i++;
        }
        delta_t = new_delta_t;

        it_count++;
    }
    std::printf("iterations: %zu\t accepted: %zu\t relation: %g\n", it_count, i, double(i) / it_count);
    return std::pair{t, y};
}

} // namespace polytrope