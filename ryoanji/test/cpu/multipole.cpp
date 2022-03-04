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

#include <iomanip>

#include "gtest/gtest.h"

#include "cstone/focus/source_center.hpp"
#include "cstone/sfc/box.hpp"
#include "coord_samples/random.hpp"
#include "ryoanji/cpu/multipole.hpp"

using namespace cstone;
using namespace ryoanji;

//! @brief Tests direct particle-to-particle gravity interactions with mass softening
TEST(Gravity, P2PmsoftBase)
{
    using T = double;

    // target
    T x = 1;
    T y = 1;
    T z = 1;
    T h = std::sqrt(3) / 2 - 0.001;

    // source
    T xs[2] = {2, -2};
    T ys[2] = {2, -2};
    T zs[2] = {2, -2};
    T hs[2] = {h, h};
    T m[2]  = {1, 1};


    auto [xacc, yacc, zacc, pot] = particle2Particle(x, y, z, h, xs, ys, zs, hs, m, 2);

    // h too small to trigger softening, so results should match the non-softened numbers
    EXPECT_NEAR(xacc, 0.17106674642655587, 1e-10);
    EXPECT_NEAR(yacc, 0.17106674642655587, 1e-10);
    EXPECT_NEAR(zacc, 0.17106674642655587, 1e-10);
    EXPECT_NEAR(pot, -0.76980035891950138, 1e-10);
}

//! @brief Tests direct particle-to-particle gravity interactions with mass softening
TEST(Gravity, P2PmsoftH)
{
    using T = double;
    constexpr int numSources = 2;

    // target
    T x = 1;
    T y = 1;
    T z = 1;
    // distance to first source is sqrt(3)/2, so here r < hi + hj
    T h = std::sqrt(3)/2 + 0.001;

    // source
    T xs[numSources] = {2, -2};
    T ys[numSources] = {2, -2};
    T zs[numSources] = {2, -2};
    T hs[numSources] = {h, h};
    T m[numSources]  = {1, 1};

    auto [xacc, yacc, zacc, pot] = particle2Particle(x, y, z, h, xs, ys, zs, hs, m, numSources);

    EXPECT_NEAR(xacc, 0.1704016164027678, 1e-10);
    EXPECT_NEAR(yacc, 0.1704016164027678, 1e-10);
    EXPECT_NEAR(zacc, 0.1704016164027678, 1e-10);
    EXPECT_NEAR(pot, -0.7678049688481372, 1e-10);
}


/*! @brief Tests the gravity interaction of a multipole with a target particle
 *
 * The gravity on the target particle is first evaluated with the direct P2P sum as a reference.
 * This is compared to the gravity on the target particle that arises from the M2P operation.
 */
TEST(Gravity, M2P)
{
    using T = double;

    cstone::Box<T> box(-1, 1);
    LocalIndex numParticles = 100;

    RandomCoordinates<T, SfcKind<unsigned>> coordinates(numParticles, box);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    std::vector<T> h(numParticles, 0);

    std::vector<T> masses(numParticles);
    std::generate(begin(masses), end(masses), drand48);

    SourceCenterType<T> center = massCenter<T>(x, y, z, masses.data(), 0, numParticles);
    CartesianQuadrupole<T> multipole;
    particle2Multipole(x, y, z, masses.data(), 0, numParticles, makeVec3(center), multipole);

    // target particle coordinates
    std::array<T, 3> target = {-8, 0, 0};

    // reference direct gravity on target
    auto [axDirect, ayDirect, azDirect, potDirect] =
        particle2Particle(target[0], target[1], target[2], 0.0, x, y, z, h.data(), masses.data(), numParticles);

    // approximate gravity with multipole interaction
    auto [axApprox, ayApprox, azApprox, potApprox] =
        multipole2Particle(target[0], target[1], target[2], makeVec3(center), multipole);

    //std::cout << std::fixed;
    //std::cout.precision(8);
    //std::cout << "direct: " << accDirect[0] << " " << accDirect[1] << " " << accDirect[2] << std::endl;
    //std::cout << "approx: " << accApprox[0] << " " << accApprox[1] << " " << accApprox[2] << std::endl;

    EXPECT_NEAR(potDirect, potApprox, 1e-3);
    EXPECT_TRUE(std::abs(axApprox - axDirect) < 1e-3);
    EXPECT_TRUE(std::abs(ayApprox - ayDirect) < 1e-3);
    EXPECT_TRUE(std::abs(azApprox - azDirect) < 1e-3);

    EXPECT_NEAR(axApprox, 0.74358243303934313,    1e-10);
    EXPECT_NEAR(ayApprox, 9.1306187450872109e-05, 1e-10);
    EXPECT_NEAR(azApprox, 0.0095252528595820823,  1e-10);
}

/*! @brief tests aggregation of multipoles into a composite multipole
 *
 * The reference multipole is directly constructed from all particles,
 * while the subcell multipoles are constructed from 1/8th of the particles each.
 * The subcells are then aggregated with the M2M operation and compared to the reference.
 */
TEST(Gravity, M2M)
{
    using T = double;

    cstone::Box<T> box(-1, 1);
    LocalIndex numParticles = 160;

    RandomCoordinates<T, SfcKind<unsigned>> coordinates(numParticles, box);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    std::vector<T> masses(numParticles);
    std::generate(begin(masses), end(masses), drand48);

    // reference directly constructed from particles
    SourceCenterType<T> refCenter = massCenter<T>(x, y, z, masses.data(), 0, numParticles);
    CartesianQuadrupole<T> reference;
    particle2Multipole(x, y, z, masses.data(), 0, numParticles, makeVec3(refCenter), reference);

    LocalIndex eighth = numParticles / 8;
    CartesianQuadrupole<T> sc[8];
    SourceCenterType<T> centers[8];
    for (int i = 0; i < 8; ++i)
    {
        centers[i] = massCenter<T>(x, y, z, masses.data(), i * eighth, (i + 1) * eighth);
        particle2Multipole(x, y, z, masses.data(), i * eighth, (i + 1) * eighth, makeVec3(centers[i]), sc[i]);
    }

    // aggregate subcell multipoles
    CartesianQuadrupole<T> composite;
    multipole2Multipole(0, 8, refCenter, centers, sc, composite);

    EXPECT_NEAR(reference[Cqi::mass], composite[Cqi::mass], 1e-10);
    EXPECT_NEAR(reference[Cqi::qxx], composite[Cqi::qxx], 1e-10);
    EXPECT_NEAR(reference[Cqi::qxy], composite[Cqi::qxy], 1e-10);
    EXPECT_NEAR(reference[Cqi::qxz], composite[Cqi::qxz], 1e-10);
    EXPECT_NEAR(reference[Cqi::qyy], composite[Cqi::qyy], 1e-10);
    EXPECT_NEAR(reference[Cqi::qyz], composite[Cqi::qyz], 1e-10);
    EXPECT_NEAR(reference[Cqi::qzz], composite[Cqi::qzz], 1e-10);
}
