/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Integration test between gravity multipole upsweep and tree walk
 *
 * @author Sebastian Keller        <sebastian.f.keller@gmail.com>
 */

#include <chrono>

#include "gtest/gtest.h"

#include "cstone/sfc/box.hpp"
#include "coord_samples/random.hpp"
#include "ryoanji/nbody/traversal_cpu.hpp"
#include "ryoanji/nbody/upsweep_cpu.hpp"

using namespace cstone;
using namespace ryoanji;

TEST(Gravity, TreeWalk)
{
    using T             = double;
    using KeyType       = uint64_t;
    using MultipoleType = ryoanji::CartesianMDQpole<T>;

    float          theta      = 0.2;
    float          G          = 1.0;
    unsigned       bucketSize = 64;
    cstone::Box<T> box(-1, 1);
    int            numShells    = 0;
    LocalIndex     numParticles = 100000;

    RandomGaussianCoordinates<T, SfcKind<KeyType>> coordinates(numParticles, box);

    const T* x = coordinates.x().data();
    const T* y = coordinates.y().data();
    const T* z = coordinates.z().data();

    std::vector<T> h(numParticles, 0.1);
    adjustSmoothingLength<KeyType>(h.size(), 2, 5, coordinates.x(), coordinates.y(), coordinates.z(), h, box);

    std::vector<T> masses(numParticles, T(1) / numParticles);
    for (size_t i = 0; i < masses.size(); ++i)
    {
        if (i % 2) masses[i] *= -1.0;
    }

    // the leaf cells and leaf particle counts
    auto [treeLeaves, counts] = computeOctree(std::span(coordinates.particleKeys()), bucketSize);

    // fully linked octree, including internal part
    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(treeLeaves));
    updateInternalTree<KeyType>(treeLeaves, octree.data());

    // layout[i] is equal to the index in (x,y,z,m) of the first particle in leaf cell with index i
    std::vector<LocalIndex> layout(octree.numLeafNodes + 1, 0);
    std::inclusive_scan(counts.begin(), counts.end(), layout.begin() + 1);

    auto toInternal = leafToInternal(octree);

    std::vector<SourceCenterType<T>> centers(octree.numNodes);
    computeLeafMassCenter<T, T, T>(coordinates.x(), coordinates.y(), coordinates.z(), masses, toInternal, layout.data(),
                                   centers.data());
    upsweep(octree.levelRange, octree.childOffsets, centers.data(), CombineSourceCenter<T>{});
    setMac<T, KeyType>(octree.prefixes, centers, 1.0 / theta, box);

    std::vector<MultipoleType> multipoles(octree.numNodes);
    computeLeafMultipoles(x, y, z, masses.data(), toInternal, layout.data(), centers.data(), multipoles.data());
    upsweepMultipoles(octree.levelRange, octree.childOffsets.data(), centers.data(), multipoles.data());

    T totalMass = std::accumulate(masses.begin(), masses.end(), 0.0);
    EXPECT_TRUE(std::abs(totalMass - multipoles[0][ryoanji::Cqi::mass]) < 1e-6);

    std::vector<T> ax(numParticles, 0);
    std::vector<T> ay(numParticles, 0);
    std::vector<T> az(numParticles, 0);

    auto t0       = std::chrono::high_resolution_clock::now();
    T    egravTot = 0;
    computeGravity(octree.childOffsets.data(), octree.internalToLeaf.data(), centers.data(), multipoles.data(),
                   layout.data(), 0, octree.numLeafNodes, x, y, z, h.data(), masses.data(), box, G, (T*)nullptr,
                   ax.data(), ay.data(), az.data(), &egravTot, numShells);
    auto   t1      = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Time elapsed for " << numParticles << " particles: " << elapsed << " s, "
              << double(numParticles) / 1e6 / elapsed << " million particles/second" << std::endl;

    // direct sum reference
    std::vector<T> Ax(numParticles, 0);
    std::vector<T> Ay(numParticles, 0);
    std::vector<T> Az(numParticles, 0);
    std::vector<T> potentialReference(numParticles, 0);

    t0 = std::chrono::high_resolution_clock::now();
    directSum(x, y, z, h.data(), masses.data(), numParticles, G, {box.lx(), box.ly(), box.lz()}, numShells, Ax.data(),
              Ay.data(), Az.data(), potentialReference.data());
    t1      = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Time elapsed for direct sum: " << elapsed << " s, " << double(numParticles) / 1e6 / elapsed
              << " million particles/second" << std::endl;

    double refPotSum = 0;
    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        refPotSum += potentialReference[i];
    }
    refPotSum *= 0.5;
    EXPECT_NEAR(std::abs(refPotSum - egravTot) / refPotSum, 0, 1e-2);

    // relative errors
    std::vector<T> delta(numParticles);
    for (LocalIndex i = 0; i < numParticles; ++i)
    {
        ryoanji::Vec3<T> axi{ax[i], ay[i], az[i]}, Axi{Ax[i], Ay[i], Az[i]};
        delta[i] = std::sqrt(norm2(axi - Axi) / norm2(Axi));
    }

    // sort errors in ascending order to infer the error distribution
    std::sort(begin(delta), end(delta));

    EXPECT_TRUE(delta[numParticles * 0.99] < 3e-3);
    EXPECT_TRUE(delta[numParticles - 1] < 3e-2);

    std::cout.precision(10);
    std::cout << "min Error: " << delta[0] << std::endl;
    // 50% of particles have an error smaller than this
    std::cout << "50th percentile: " << delta[numParticles / 2] << std::endl;
    // 90% of particles have an error smaller than this
    std::cout << "10th percentile: " << delta[numParticles * 0.9] << std::endl;
    // 99% of particles have an error smaller than this
    std::cout << "1st percentile: " << delta[numParticles * 0.99] << std::endl;
    std::cout << "max Error: " << delta[numParticles - 1] << std::endl;
}
