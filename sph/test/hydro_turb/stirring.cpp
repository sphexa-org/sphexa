
#include <vector>

#include "gtest/gtest.h"

#include "sph/hydro_turb/stirring.hpp"
#include "sph/hydro_turb/phases.hpp"

/*! @brief Test stirring forces due to given modes
 *
 * 1. define an (x,y,z) particle
 * 2. create a few modes, amplitudes and phases
 * 3. apply stirring and check resulting accelerations
 */
TEST(Turbulence, stirParticle)
{
    using T = double;

    T      tol  = 1.e-8;
    size_t ndim = 3;

    T xi{0.5}, yi{0.5}, zi{0.5};

    int numModes      = 1;
    T   modes[3]      = {12.566370614359172, 0, 0}; // 2*k = 4*pi
    T   phaseReal[3]  = {1.0, 1.0, 1.0};
    T   phaseImag[3]  = {1.0, 1.0, 1.0};
    T   amplitudes[1] = {2.0};

    auto [ax, ay, az] = sph::stirParticle<T, T, T>(ndim, xi, yi, zi, numModes, modes, phaseReal, phaseImag, amplitudes);

    // analytic value is 2.0, fortran code somehow is less accurate (1.9999996503088493)
    EXPECT_NEAR(ax, 2.0, tol);
    EXPECT_NEAR(ay, 2.0, tol);
    EXPECT_NEAR(az, 2.0, tol);
}

TEST(Turbulence, stirParticle2)
{
    using T = double;

    T      tol  = 3.e-7;
    size_t ndim = 3;

    T xi{-0.2}, yi{0.3}, zi{-0.4};

    int numModes      = 1;
    T   modes[3]      = {10.0, 15.0, 14.0};
    T   phaseReal[3]  = {1.1, 1.2, 1.3};
    T   phaseImag[3]  = {0.7, 0.8, 0.9};
    T   amplitudes[1] = {2.0};

    auto [ax, ay, az] = sph::stirParticle<T, T, T>(ndim, xi, yi, zi, numModes, modes, phaseReal, phaseImag, amplitudes);

    EXPECT_NEAR(ax, -2.1398843541189447, tol);
    EXPECT_NEAR(ay, -2.3313952836997660, tol);
    EXPECT_NEAR(az, -2.5229059800250133, tol);
}

TEST(Turbulence, stirParticle3)
{
    using T = double;

    T      tol  = 5.e-7;
    size_t ndim = 3;

    T xi{-0.2}, yi{0.3}, zi{-0.4};

    int numModes      = 2;
    T   modes[6]      = {10.0, 15.0, 14.0, 12.566370614359172, 0, 0};
    T   phaseReal[6]  = {1.1, 1.2, 1.3, 1.0, 1.0, 1.0};
    T   phaseImag[6]  = {0.7, 0.8, 0.9, 1.0, 1.0, 1.0};
    T   amplitudes[2] = {2.0, 1.0};

    auto [ax, ay, az] = sph::stirParticle<T, T, T>(ndim, xi, yi, zi, numModes, modes, phaseReal, phaseImag, amplitudes);

    EXPECT_NEAR(ax, -2.1398843541189447 - 0.22123189208356875, tol);
    EXPECT_NEAR(ay, -2.3313952836997660 - 0.22123189208356875, tol);
    EXPECT_NEAR(az, -2.5229059800250133 - 0.22123189208356875, tol);
}

TEST(Turbulence, computePhases)
{
    using T = double;

    T              tol        = 1.e-8;
    size_t         ndim       = 3;
    size_t         numModes   = 1;
    std::vector<T> modes      = {10.0, 15.0, 14.0};
    std::vector<T> phasesReal = {0, 0, 0};
    std::vector<T> phasesImag = {0, 0, 0};
    std::vector<T> OUPhases   = {0.7, 0.8, 0.9, 1.0, 1.0, 1.0};
    T              solWeight  = 0.5;

    sph::computePhases(numModes, ndim, OUPhases, solWeight, modes, phasesReal, phasesImag);

    // comparison against analytical values
    EXPECT_NEAR(phasesReal[0], 0.35, tol); // fortran value 0.34999999403953552
    EXPECT_NEAR(phasesReal[1], 0.45, tol); // fortran value 0.44999998807907104
    EXPECT_NEAR(phasesReal[2], 0.50, tol); // fortran value 0.50000000000000000
    EXPECT_NEAR(phasesImag[0], 0.40, tol); // fortran value 0.40000000596046448
    EXPECT_NEAR(phasesImag[1], 0.50, tol); // fortran value 0.50000000000000000
    EXPECT_NEAR(phasesImag[2], 0.50, tol); // fortran value 0.50000000000000000
}
