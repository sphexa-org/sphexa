//
// Created by Noah Kubli on 18.03.2025.
//

#pragma once

#include <cmath>
#include <utility>

namespace polytrope
{
template<typename F>
std::pair<bool, double> find_zero_bisect(F&& f, double min, double max, const double tolerance,
                                         const double r_tolerance)
{
    while (true)
    {
        double mean   = (max + min) / 2.;
        double f_mean = f(mean);
        if (std::abs(f_mean) < tolerance) return {true, mean};
        if (std::abs(max - min) < r_tolerance) return {true, mean};
        double f_min = f(min);
        double f_max = f(max);
        if ((f_min > 0. && f_max > 0.) || (f_min < 0. && f_max < 0.))
            return {false, 0.};
        if (f_min > 0.)
        {
            if (f_mean > 0.) { min = mean; }
            else { max = mean; }
        }
        else if (f_max > 0.)
        {
            if (f_mean > 0.) { max = mean; }
            else { min = mean; }
        }
    }
}
}; // namespace polytrope
