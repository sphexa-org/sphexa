#pragma once

#include <map>
#include <string>

#include "init/settings.hpp"

namespace sphexa
{

InitSettings sedovConstants()
{
    InitSettings ret{{"dim", 3},   {"gamma", 5. / 3.}, {"omega", 0.},       {"r0", 0.},
                     {"r1", 0.5},  {"mTotal", 1.},     {"energyTotal", 1.}, {"width", 0.1},
                     {"rho0", 1.}, {"u0", 1e-8},       {"p0", 0.},          {"vr0", 0.},
                     {"cs0", 0.},  {"minDt", 1e-8},    {"minDt_m1", 1e-8},  {"gravConstant", 0.0},
                     {"ng0", 100}, {"ngmax", 150},     {"mui", 10}};

    ret["ener0"] = ret["energyTotal"] / std::pow(M_PI, 1.5) / 1. / std::pow(ret["width"], 3.0);
    return ret;
}

InitSettings magneticSedovConstants()
{
    InitSettings ret{{"dim", 3},   {"gamma", 5. / 3.}, {"omega", 0.},       {"r0", 0.},
                     {"r1", 0.5},  {"mTotal", 1.},     {"energyTotal", 5.}, {"width", 0.1},
                     {"rho0", 1.}, {"u0", 1.},         {"p0", 0.},          {"vr0", 0.},
                     {"cs0", 0.},  {"minDt", 1e-8},    {"minDt_m1", 1e-8},  {"gravConstant", 0.0},
                     {"ng0", 100}, {"ngmax", 150},     {"mui", 10},         {"Bmag", 10}};

    ret["ener0"] = ret["energyTotal"] / std::pow(M_PI, 1.5) / 1. / std::pow(ret["width"], 3.0);
    return ret;
}

} // namespace sphexa
