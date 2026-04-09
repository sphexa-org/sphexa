#pragma once

#include <map>
#include <string>

#include "init/settings.hpp"

namespace sphexa
{

InitSettings sedovConstants()
{
    InitSettings ret{{"dim", 3},   {"gamma", 5. / 3.}, {"omega", 0.},       {"r0", 0.},
                     {"r1", 0.5},  {"mTotal", 1.},     {"energyTotal", 1.}, 
                     {"rho0", 1.}, {"u0", 1e-12},      {"p0", 0.},          {"vr0", 0.},
                     {"cs0", 0.},  {"minDt", 1e-6},    {"minDt_m1", 1e-6},  {"gravConstant", 0.0},
                     {"ng0", 100}, {"ngmax", 150},     {"mui", 10},         {"Kcour", 0.2}};

    return ret;
}

} // namespace sphexa
