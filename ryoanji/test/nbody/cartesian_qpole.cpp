/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

#include <iomanip>

#include "gtest/gtest.h"

#include "cstone/focus/source_center.hpp"
#include "coord_samples/random.hpp"
#include "ryoanji/nbody/cartesian_qpole.hpp"
#include "ryoanji/nbody/kernel.hpp"

using namespace cstone;
using namespace ryoanji;

/*! @brief Tests the gravity interaction of a multipole with a target particle
 *
 * The gravity on the target particle is first evaluated with the direct P2P sum as a reference.
 * This is compared to the gravity on the target particle that arises from the M2P operation.
 */
TEST(Gravity, Cartesian_M2P)
{
    using T = double;

    cstone::Box<T> box(-1, 1);
    LocalIndex     numParticles = 100;

    RandomCoordinates<T, SfcKind<unsigned>> coordinates(numParticles, box);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    std::vector<T> h(numParticles, 0);

    std::vector<T> masses(numParticles);
    std::generate(begin(masses), end(masses), drand48);

    SourceCenterType<T>    center = massCenter<T>(x, y, z, masses.data(), 0, numParticles);
    CartesianQuadrupole<T> multipole;
    P2M(x, y, z, masses.data(), 0, numParticles, center, multipole);

    // target particle coordinates
    util::array<T, 3> target{-8, 0, 0};

    // reference direct gravity on target
    cstone::Vec4<T> accP2P{0, 0, 0, 0};
    for (unsigned s = 0; s < numParticles; ++s)
    {
        accP2P = P2P(accP2P, target, cstone::Vec3<T>{x[s], y[s], z[s]}, masses[s], 0.0, h[s]);
    }

    // approximate gravity with multipole interaction
    cstone::Vec4<T> accApprox{0, 0, 0, 0};
    accApprox = M2P(accApprox, target, makeVec3(center), multipole);

    // std::cout << std::fixed;
    // std::cout.precision(8);
    // std::cout << "direct: " << accDirect[0] << " " << accDirect[1] << " " << accDirect[2] << std::endl;
    // std::cout << "approx: " << accApprox[0] << " " << accApprox[1] << " " << accApprox[2] << std::endl;

    EXPECT_NEAR(accP2P[0], accApprox[0], 1e-3);
    EXPECT_TRUE(std::abs(accApprox[1] - accP2P[1]) < 1e-3);
    EXPECT_TRUE(std::abs(accApprox[2] - accP2P[2]) < 1e-3);
    EXPECT_TRUE(std::abs(accApprox[3] - accP2P[3]) < 1e-3);

    EXPECT_NEAR(accApprox[1], 0.74358243303934313, 1e-10);
    EXPECT_NEAR(accApprox[2], 9.1306187450872109e-05, 1e-10);
    EXPECT_NEAR(accApprox[3], 0.0095252528595820823, 1e-10);
}

/*! @brief tests aggregation of multipoles into a composite multipole
 *
 * The reference multipole is directly constructed from all particles,
 * while the subcell multipoles are constructed from 1/8th of the particles each.
 * The subcells are then aggregated with the M2M operation and compared to the reference.
 */
TEST(Gravity, Cartesian_M2M)
{
    using T = double;

    cstone::Box<T> box(-1, 1);
    LocalIndex     numParticles = 160;

    RandomCoordinates<T, SfcKind<unsigned>> coordinates(numParticles, box);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    std::vector<T> masses(numParticles);
    std::generate(begin(masses), end(masses), drand48);

    // reference directly constructed from particles
    SourceCenterType<T>    refCenter = massCenter<T>(x, y, z, masses.data(), 0, numParticles);
    CartesianQuadrupole<T> reference;
    P2M(x, y, z, masses.data(), 0, numParticles, refCenter, reference);

    LocalIndex             eighth = numParticles / 8;
    CartesianQuadrupole<T> sc[8];
    SourceCenterType<T>    centers[8];
    for (int i = 0; i < 8; ++i)
    {
        centers[i] = massCenter<T>(x, y, z, masses.data(), i * eighth, (i + 1) * eighth);
        P2M(x, y, z, masses.data(), i * eighth, (i + 1) * eighth, centers[i], sc[i]);
    }

    // aggregate subcell multipoles
    CartesianQuadrupole<T> composite;
    M2M(0, 8, refCenter, centers, sc, composite);

    EXPECT_NEAR(reference[Cqi::mass], composite[Cqi::mass], 1e-10);
    EXPECT_NEAR(reference[Cqi::qxx], composite[Cqi::qxx], 1e-10);
    EXPECT_NEAR(reference[Cqi::qxy], composite[Cqi::qxy], 1e-10);
    EXPECT_NEAR(reference[Cqi::qxz], composite[Cqi::qxz], 1e-10);
    EXPECT_NEAR(reference[Cqi::qyy], composite[Cqi::qyy], 1e-10);
    EXPECT_NEAR(reference[Cqi::qyz], composite[Cqi::qyz], 1e-10);
    EXPECT_NEAR(reference[Cqi::qzz], composite[Cqi::qzz], 1e-10);
}

TEST(Gravity, CartesianMDQ_M2P)
{
    using T = double;

    cstone::Box<T> box(-1, 1);
    LocalIndex     numParticles = 100;

    RandomCoordinates<T, SfcKind<unsigned>> coordinates(numParticles, box);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    std::vector<T> h(numParticles, 0);

    std::vector<T> masses(numParticles);
    std::generate(begin(masses), end(masses), []() { return drand48() - 0.5; });

    SourceCenterType<T> center = massCenter<T>(x, y, z, masses.data(), 0, numParticles);
    CartesianMDQpole<T> multipole;
    P2M(x, y, z, masses.data(), 0, numParticles, center, multipole);

    // target particle coordinates
    util::array<T, 3> target{-8, 0, 0};

    // reference direct gravity on target
    cstone::Vec4<T> accP2P{0, 0, 0, 0};
    for (unsigned s = 0; s < numParticles; ++s)
    {
        accP2P = P2P(accP2P, target, cstone::Vec3<T>{x[s], y[s], z[s]}, masses[s], 0.0, h[s]);
    }

    // approximate gravity with multipole interaction
    cstone::Vec4<T> accApprox{0, 0, 0, 0};
    accApprox = M2P(accApprox, target, makeVec3(center), multipole);

    std::cout << std::fixed;
    std::cout.precision(8);
    std::cout << "direct: " << accP2P[1] << " " << accP2P[2] << " " << accP2P[3] << std::endl;
    std::cout << "approx: " << accApprox[1] << " " << accApprox[2] << " " << accApprox[3] << std::endl;

    EXPECT_NEAR(accP2P[0], accApprox[0], 1e-3);
    EXPECT_TRUE(std::abs(accApprox[1] - accP2P[1]) < 1e-3);
    EXPECT_TRUE(std::abs(accApprox[2] - accP2P[2]) < 1e-3);
    EXPECT_TRUE(std::abs(accApprox[3] - accP2P[3]) < 1e-3);
}

/*! @brief tests aggregation of multipoles into a composite multipole
 *
 * The reference multipole is directly constructed from all particles,
 * while the subcell multipoles are constructed from 1/8th of the particles each.
 * The subcells are then aggregated with the M2M operation and compared to the reference.
 */
TEST(Gravity, CartesianMDQ_M2M)
{
    using T = double;

    cstone::Box<T> box(0, 1);
    LocalIndex     numParticles = 160;

    RandomCoordinates<T, SfcKind<unsigned>> coordinates(numParticles, box);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    std::vector<T> masses(numParticles);
    std::generate(begin(masses), end(masses), []() { return drand48() - 0.5; });

    // reference directly constructed from particles
    SourceCenterType<T> refCenter = massCenter<T>(x, y, z, masses.data(), 0, numParticles);
    CartesianMDQpole<T> reference;
    P2M(x, y, z, masses.data(), 0, numParticles, refCenter, reference);

    LocalIndex          eighth = numParticles / 8;
    CartesianMDQpole<T> sc[8];
    SourceCenterType<T> centers[8];
    for (int i = 0; i < 8; ++i)
    {
        centers[i] = massCenter<T>(x, y, z, masses.data(), i * eighth, (i + 1) * eighth);
        P2M(x, y, z, masses.data(), i * eighth, (i + 1) * eighth, centers[i], sc[i]);
    }

    // aggregate subcell multipoles
    CartesianMDQpole<T> composite;
    M2M(0, 8, refCenter, centers, sc, composite);

    EXPECT_NEAR(reference[Cqi::mass], composite[Cqi::mass], 1e-10);
    EXPECT_NEAR(reference[Cqi::qxx], composite[Cqi::qxx], 1e-10);
    EXPECT_NEAR(reference[Cqi::qxy], composite[Cqi::qxy], 1e-10);
    EXPECT_NEAR(reference[Cqi::qxz], composite[Cqi::qxz], 1e-10);
    EXPECT_NEAR(reference[Cqi::qyy], composite[Cqi::qyy], 1e-10);
    EXPECT_NEAR(reference[Cqi::qyz], composite[Cqi::qyz], 1e-10);
    EXPECT_NEAR(reference[Cqi::qzz], composite[Cqi::qzz], 1e-10);
    EXPECT_NEAR(reference[Cqi::px], composite[Cqi::px], 1e-10);
    EXPECT_NEAR(reference[Cqi::py], composite[Cqi::py], 1e-10);
    EXPECT_NEAR(reference[Cqi::pz], composite[Cqi::pz], 1e-10);

    // compare accelerations by reference and composite multipole on a test target
    util::array<T, 3> target{-8, 0, 0};

    cstone::Vec4<T> accMRef{0, 0, 0, 0};
    accMRef = M2P(accMRef, target, makeVec3(refCenter), reference);
    cstone::Vec4<T> accMComp{0, 0, 0, 0};
    accMComp = M2P(accMComp, target, makeVec3(refCenter), composite);

    std::cout << std::fixed;
    std::cout.precision(8);
    std::cout << "ref : " << accMRef[1] << " " << accMRef[2] << " " << accMRef[3] << std::endl;
    std::cout << "comp: " << accMComp[1] << " " << accMComp[2] << " " << accMComp[3] << std::endl;
    EXPECT_NEAR(std::sqrt(norm2(accMRef - accMComp)), 0.0, 1e-3);
}
