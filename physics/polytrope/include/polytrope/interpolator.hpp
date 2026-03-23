//
//  interpolator.hpp
//  Polytrop
//
//  Created by Noah Kubli on 15.02.2025.
//

#pragma once

#include <algorithm>
#include <vector>

namespace polytrope
{
struct LinearInterpolator
{
    std::vector<double> x_values;
    std::vector<double> y_values;
    double              operator()(const double x) const
    {
        const auto it = std::upper_bound(x_values.begin(), x_values.end(), x);
        if (it == x_values.end()) return y_values.back();
        if (it == x_values.begin()) return y_values.front();

        const size_t n        = it - x_values.begin();
        const double x_lower  = x_values[n - 1];
        const double x_higher = x_values[n];
        const double t        = (x - x_lower) / (x_higher - x_lower);

        const double y_lower  = y_values[n - 1];
        const double y_higher = y_values[n];

        return t * y_higher + (1. - t) * y_lower;
    }
};
} // namespace polytrope
