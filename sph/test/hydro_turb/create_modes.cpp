#include <map>
#include <string>

#include "gtest/gtest.h"

#include "sph/hydro_turb/create_modes.hpp"

std::map<std::string, double> TurbulenceConstants()
{
    return {{"solWeight", 0.5},        {"stMaxModes", 100000}, {"Lbox", 1.0},       {"stEnergyPrefac", 5.0e-3},
            {"stMachVelocity", 0.3e0}, {"epsilon", 1e-15},     {"rngSeed", 251299}, {"stSpectForm", 1},
            {"powerLawExp", 5. / 3},   {"anglesExp", 2.0}};
};

TEST(Turbulence, spectForm1_unifdimensions)
{
    using T = double;

    const T twopi = 6.283185307179586;
    const T tol   = 1.e-6;

    T      Lx          = 1.0;
    T      Ly          = 1.0;
    T      Lz          = 1.0;
    T      stirMax     = 3.000000000000000 * twopi / Lx;
    T      stirMin     = (1.0) * twopi / Lx;
    size_t st_maxmodes = 100000;
    size_t ndim        = 3;
    int    numModes    = 0;
    // T      modes[st_maxmodes];
    // T      amplitudes[st_maxmodes];
    bool verbose = true;

    sph::TurbulenceData<T, cstone::CpuTag> turbulenceData(TurbulenceConstants(), verbose);
    sph::createStirringModes(turbulenceData, Lx, Ly, Lz, st_maxmodes, stirMax, stirMin, ndim, 1, 0.0, 0.0, verbose);

    numModes         = turbulenceData.numModes;
    auto& modes      = turbulenceData.modes;
    auto& amplitudes = turbulenceData.amplitudes;

    EXPECT_EQ(numModes, 112);

    EXPECT_NEAR(modes[3 * (10 - 1) + 0], 0.0000000000000000, tol);
    EXPECT_NEAR(modes[3 * (10 - 1) + 1], -0.0000000000000000, tol);
    EXPECT_NEAR(modes[3 * (10 - 1) + 2], 18.849556446075439, tol);
    EXPECT_NEAR(0.5 * amplitudes[(10 - 1)], 0.0000000000000000, tol);

    EXPECT_NEAR(modes[3 * (54 - 1) + 0], 6.2831854820251465, tol);
    EXPECT_NEAR(modes[3 * (54 - 1) + 1], -6.2831854820251465, tol);
    EXPECT_NEAR(modes[3 * (54 - 1) + 2], 0.0000000000000000, tol);
    EXPECT_NEAR(0.5 * amplitudes[(54 - 1)], 1.1461712345826693, tol);

    EXPECT_NEAR(modes[3 * (78 - 1) + 0], 12.566370964050293, tol);
    EXPECT_NEAR(modes[3 * (78 - 1) + 1], -0.0000000000000000, tol);
    EXPECT_NEAR(modes[3 * (78 - 1) + 2], 0.0000000000000000, tol);
    EXPECT_NEAR(0.5 * amplitudes[(78 - 1)], 1.0000000000000000, tol);
}

TEST(Turbulence, spectForm1_nonunifdimensions)
{
    using T = double;

    const T twopi       = 2.0 * M_PI;
    const T tol         = 1.e-6;
    T       Lx          = 0.7;
    T       Ly          = 1.2;
    T       Lz          = 1.5;
    T       stirMax     = 3.000000000000001 * twopi / Lx;
    T       stirMin     = (0.999999999999999) * twopi / Lx;
    size_t  st_maxmodes = 100000;
    size_t  ndim        = 3;
    int     numModes    = 0;
    // T modes[st_maxmodes];
    // T amplitudes[st_maxmodes];
    bool verbose = true;

    sph::TurbulenceData<T, cstone::CpuTag> turbulenceData(TurbulenceConstants(), verbose);
    sph::createStirringModes(turbulenceData, Lx, Ly, Lz, st_maxmodes, stirMax, stirMin, ndim, 1, 0.0, 0.0, verbose);

    numModes         = turbulenceData.numModes;
    auto& modes      = turbulenceData.modes;
    auto& amplitudes = turbulenceData.amplitudes;

    EXPECT_EQ(numModes, 300);

    EXPECT_NEAR(modes[3 * (10 - 1) + 0], 0.0000000000000000, tol);
    EXPECT_NEAR(modes[3 * (10 - 1) + 1], -0.0000000000000000, tol);
    EXPECT_NEAR(modes[3 * (10 - 1) + 2], 20.943951606750488, tol);
    EXPECT_NEAR(0.5 * amplitudes[(10 - 1)], 0.80812206144596110, tol);

    EXPECT_NEAR(modes[3 * (54 - 1) + 0], 0.0000000000000000, tol);
    EXPECT_NEAR(modes[3 * (54 - 1) + 1], -10.471975387256329, tol);
    EXPECT_NEAR(modes[3 * (54 - 1) + 2], 16.755161285400391, tol);
    EXPECT_NEAR(0.5 * amplitudes[(54 - 1)], 0.88997794370873951, tol);

    EXPECT_NEAR(modes[3 * (78 - 1) + 0], 0.0000000000000000, tol);
    EXPECT_NEAR(modes[3 * (78 - 1) + 1], -15.707963080884493, tol);
    EXPECT_NEAR(modes[3 * (78 - 1) + 2], 16.755161285400391, tol);
    EXPECT_NEAR(0.5 * amplitudes[(78 - 1)], 0.64827464011102576, tol);
}
