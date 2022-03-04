/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Compare and test different multipole approximations
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/focus/source_center.hpp"

#include "dataset.hpp"
#include "ryoanji/cpu/multipole.hpp"
#include "ryoanji/kernel.hpp"

using namespace ryoanji;

TEST(Multipole, P2M)
{
    int numBodies = 1023;

    std::vector<Vec4<float>> bodies(numBodies);
    ryoanji::makeCubeBodies(bodies.data(), numBodies);

    std::vector<double> x(numBodies);
    std::vector<double> y(numBodies);
    std::vector<double> z(numBodies);
    std::vector<double> h(numBodies, 0.0);
    std::vector<double> m(numBodies);

    for (size_t i = 0; i < numBodies; ++i)
    {
        x[i] = bodies[i][0];
        y[i] = bodies[i][1];
        z[i] = bodies[i][2];
        m[i] = bodies[i][3];
    }

    CartesianQuadrupole<double>      cartesianQuadrupole;
    cstone::SourceCenterType<double> csCenter =
        cstone::massCenter<double>(x.data(), y.data(), z.data(), m.data(), 0, numBodies);
    particle2Multipole(
        x.data(), y.data(), z.data(), m.data(), 0, numBodies, util::makeVec3(csCenter), cartesianQuadrupole);

    Vec4<float> centerMass = ryoanji::setCenter(0, numBodies, bodies.data());

    ryoanji::SphericalMultipole<float, 4> sphericalOctopole;
    std::fill(sphericalOctopole.begin(), sphericalOctopole.end(), 0.0);

    ryoanji::P2M(0, numBodies, centerMass, bodies.data(), sphericalOctopole);

    EXPECT_NEAR(sphericalOctopole[0], cartesianQuadrupole[Cqi::mass], 1e-6);

    EXPECT_NEAR(centerMass[0], csCenter[0] , 1e-6);
    EXPECT_NEAR(centerMass[1], csCenter[1] , 1e-6);
    EXPECT_NEAR(centerMass[2], csCenter[2] , 1e-6);
    EXPECT_NEAR(centerMass[3], cartesianQuadrupole[Cqi::mass], 1e-6);

    // compare M2P results on a test target
    {
        float eps2 = 0;
        Vec3<float> testTarget{-8, -8, -8};

        Vec4<float> acc{0, 0, 0, 0};
        acc = ryoanji::M2P(acc, testTarget, util::makeVec3(centerMass), sphericalOctopole, eps2);
        //printf("test acceleration: %f %f %f %f\n", acc[0], acc[1], acc[2], acc[3]);

        // cstone is less precise
        //float ax = 0;
        //float ay = 0;
        //float az = 0;
        //cstone::multipole2particle(
        //    testTarget[0], testTarget[1], testTarget[2], cstoneMultipole, eps2, &ax, &ay, &az);
        //printf("cstone test acceleration: %f %f %f\n", ax, ay, az);

        auto [axd, ayd, azd, pot] = particle2Particle(double(testTarget[0]),
                                                      double(testTarget[1]),
                                                      double(testTarget[2]),
                                                      0.0,
                                                      x.data(),
                                                      y.data(),
                                                      z.data(),
                                                      h.data(),
                                                      m.data(),
                                                      numBodies);
        // printf("direct acceleration: %f %f %f\n", axd, ayd, azd);

        // compare ryoanji against the direct sum reference
        EXPECT_NEAR(acc[0], pot, 3e-5);
        EXPECT_NEAR(acc[1], axd, 1e-5);
        EXPECT_NEAR(acc[2], ayd, 1e-5);
        EXPECT_NEAR(acc[3], azd, 1e-5);
    }
}

