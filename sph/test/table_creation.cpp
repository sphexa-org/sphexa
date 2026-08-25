
/*! @file
 * @brief test 1D simpson numerical integrator
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <cmath>
#include <cstdio>

#include "gtest/gtest.h"

#include "sph/sph_kernels.hpp"

using namespace sph;

TEST(KernelTable, simpsonSine)
{
    auto sin = [](double x) { return std::sin(x); };
    // integrate sin(x) from 0 to PI/2
    double integral = util::simpson(0, M_PI / 2, 2000, sin);
    // 2000 intervals yield and accuracy of 1e-14
    EXPECT_NEAR(integral, 1.0, 1e-14);
}

TEST(KernelTable, simpson3DK)
{
    double n  = 6;
    auto   Sn = [n](double x)
    {
        if (x == 0.0) { return 0.0; }
        auto Pv = M_PI_2 * x;
        return std::pow(std::sin(Pv) / Pv, n);
    };

    double Bn = kernel_3D_k(Sn);

    printf("3D-K: interpolated %.16f, integrated %.16f, diff %.16f\n", sphynx_3D_k(n), Bn, sphynx_3D_k(n) - Bn);
    EXPECT_NEAR(sphynx_3D_k(n), Bn, 1e-4);
}
