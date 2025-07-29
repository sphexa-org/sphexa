/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Single-GPU demonstrator app for the Ryoanji N-body library
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Rio Yokota <rioyokota@gsic.titech.ac.jp>
 */

#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/gpu_config.cuh"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/focus/source_center_gpu.h"
#include "cstone/traversal/groups_gpu.h"
#include "cstone/util/array.hpp"

#include "nbody/dataset.hpp"
#include "ryoanji/interface/treebuilder.cuh"
#include "ryoanji/nbody/types.h"
#include "ryoanji/nbody/cartesian_qpole.hpp"
#include "ryoanji/nbody/traversal_gpu.h"
#include "ryoanji/nbody/direct.cuh"
#include "ryoanji/nbody/upsweep_gpu.h"

using namespace ryoanji;

template<class Tc, class Th, class Tm, class Ta, class Tf, class MType>
util::array<Tc, 5> computeAcceleration(size_t firstBody, size_t lastBody, const Tc* x, const Tc* y, const Tc* z,
                                       const Tm* m, const Th* h, Tc G, int numShells, const cstone::Box<Tc>& box, Ta* p,
                                       Ta* ax, Tc* ay, Tc* az, const TreeNodeIndex* childOffsets,
                                       const TreeNodeIndex* internalToLeaf, const LocalIndex* layout,
                                       const Vec4<Tf>* sourceCenter, const MType* Multipole);

template<class KeyType, class T, class MType>
void upsweep(int numSources, int numLeaves, int numLevels, float theta, const TreeNodeIndex* levelRange, const T* x,
             const T* y, const T* z, const T* m, const cstone::Box<T>& box, const LocalIndex* layout,
             const KeyType* prefixes, const TreeNodeIndex* childOffsets, const TreeNodeIndex* leafToInternal,
             Vec4<T>* centers, MType* Multipole);

int main(int argc, char** argv)
{
    using T             = float;
    using MultipoleType = CartesianQuadrupole<T>;

    int power     = argc > 1 ? std::stoi(argv[1]) : 17;
    int directRef = argc > 2 ? std::stoi(argv[2]) : 1;
    int numShells = argc > 3 ? std::stoi(argv[3]) : 0;

    std::size_t numBodies = (1 << power) - 1;
    T           theta     = 0.6;
    T           boxSize   = 3;
    T           G         = 1.0;

    const int ncrit = 64;

    fprintf(stdout, "--- BH Parameters ---------------\n");
    fprintf(stdout, "numBodies            : %lu\n", numBodies);
    fprintf(stdout, "theta                : %f\n", theta);
    fprintf(stdout, "ncrit                : %d\n", ncrit);

    thrust::host_vector<T> x(numBodies), y(numBodies), z(numBodies), m(numBodies), h(numBodies);
    makeCubeBodies(x.data(), y.data(), z.data(), m.data(), h.data(), numBodies, boxSize);

    // upload bodies to device
    thrust::device_vector<T> d_x = x, d_y = y, d_z = z, d_m = m, d_h = h;

    cstone::Box<T> box(-boxSize, boxSize);

    TreeBuilder<uint64_t> treeBuilder(ncrit);
    int                   numSources = treeBuilder.update(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), numBodies, box);

    const TreeNodeIndex* levelRange   = treeBuilder.levelRange();
    int                  highestLevel = treeBuilder.maxTreeLevel();

    thrust::device_vector<Vec4<T>>       sourceCenter(numSources);
    thrust::device_vector<MultipoleType> Multipole(numSources);

    upsweep(numSources, treeBuilder.numLeafNodes(), highestLevel, theta, levelRange, rawPtr(d_x), rawPtr(d_y),
            rawPtr(d_z), rawPtr(d_m), box, treeBuilder.layout(), treeBuilder.nodeKeys(), treeBuilder.childOffsets(),
            treeBuilder.leafToInternal(), rawPtr(sourceCenter), rawPtr(Multipole));

    thrust::device_vector<T> d_p(numBodies, 0), d_ax(numBodies, 0), d_ay(numBodies, 0), d_az(numBodies, 0);

    fprintf(stdout, "--- BH Profiling ----------------\n");

    auto t0 = std::chrono::high_resolution_clock::now();

    auto interactions = computeAcceleration(0, numBodies, rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m),
                                            rawPtr(d_h), G, numShells, box, rawPtr(d_p), rawPtr(d_ax), rawPtr(d_ay),
                                            rawPtr(d_az), treeBuilder.childOffsets(), treeBuilder.internalToLeaf(),
                                            treeBuilder.layout(), rawPtr(sourceCenter), rawPtr(Multipole));

    auto   t1    = std::chrono::high_resolution_clock::now();
    double dt    = std::chrono::duration<double>(t1 - t0).count();
    double flops = (interactions[0] * 23 + interactions[2] * 65) * numBodies / dt / 1e12;

    fprintf(stdout, "--- Total runtime ----------------\n");
    fprintf(stdout, "Total BH            : %.7f s (%.7f TFlops)\n", dt, flops);

    if (!directRef) { return 0; }

    thrust::device_vector<T> refP(numBodies), refAx(numBodies), refAy(numBodies), refAz(numBodies);

    t0 = std::chrono::high_resolution_clock::now();
    directSum(0, numBodies, numBodies, Vec3<T>{box.lx(), box.ly(), box.lz()}, numShells, rawPtr(d_x), rawPtr(d_y),
              rawPtr(d_z), rawPtr(d_m), rawPtr(d_h), rawPtr(refP), rawPtr(refAx), rawPtr(refAy), rawPtr(refAz));

    t1 = std::chrono::high_resolution_clock::now();
    dt = std::chrono::duration<double>(t1 - t0).count();

    flops = std::pow((2 * numShells + 1), 3) * 23. * numBodies * numBodies / dt / 1e12;
    fprintf(stdout, "Total Direct         : %.7f s (%.7f TFlops)\n", dt, flops);

    thrust::host_vector<T> h_p  = d_p;
    thrust::host_vector<T> h_ax = d_ax;
    thrust::host_vector<T> h_ay = d_ay;
    thrust::host_vector<T> h_az = d_az;

    double                 referencePotential = 0.5 * G * thrust::reduce(refP.begin(), refP.end(), 0.0);
    thrust::host_vector<T> h_refAx            = refAx;
    thrust::host_vector<T> h_refAy            = refAy;
    thrust::host_vector<T> h_refAz            = refAz;

    std::vector<double> delta(numBodies);

    double potentialSum = 0;
    for (int i = 0; i < numBodies; i++)
    {
        potentialSum += h_p[i];
        Vec3<T> ref   = {h_refAx[i], h_refAy[i], h_refAz[i]};
        Vec3<T> probe = {h_ax[i], h_ay[i], h_az[i]};
        delta[i]      = std::sqrt(norm2(ref - probe) / norm2(ref));
    }

    std::sort(begin(delta), end(delta));

    fprintf(stdout, "--- BH vs. direct ---------------\n");

    std::cout << "potentials, body-sum: " << 0.5 * G * potentialSum << " atomic sum: " << interactions[4]
              << " reference: " << referencePotential << std::endl;
    std::cout << "min Error: " << delta[0] << std::endl;
    std::cout << "50th percentile: " << delta[numBodies / 2] << std::endl;
    std::cout << "10th percentile: " << delta[numBodies * 0.9] << std::endl;
    std::cout << "1st percentile: " << delta[numBodies * 0.99] << std::endl;
    std::cout << "max Error: " << delta[numBodies - 1] << std::endl;

    fprintf(stdout, "--- Tree stats -------------------\n");
    fprintf(stdout, "Bodies               : %lu\n", numBodies);
    fprintf(stdout, "Cells                : %d\n", numSources);
    fprintf(stdout, "Tree depth           : %d\n", highestLevel);
    fprintf(stdout, "--- Traversal stats --------------\n");
    fprintf(stdout, "P2P mean list length : %d (max %d)\n", int(interactions[0]), int(interactions[1]));
    fprintf(stdout, "M2P mean list length : %d (max %d)\n", int(interactions[2]), int(interactions[3]));

    return 0;
}

template<class Tc, class Th, class Tm, class Ta, class Tf, class MType>
util::array<Tc, 5> computeAcceleration(size_t firstBody, size_t lastBody, const Tc* x, const Tc* y, const Tc* z,
                                       const Tm* m, const Th* h, Tc G, int numShells, const cstone::Box<Tc>& box, Ta* p,
                                       Ta* ax, Tc* ay, Tc* az, const TreeNodeIndex* childOffsets,
                                       const TreeNodeIndex* internalToLeaf, const LocalIndex* layout,
                                       const Vec4<Tf>* sourceCenter, const MType* Multipole)
{
    auto                              numBodies = lastBody - firstBody;
    cstone::GroupData<cstone::GpuTag> groups;
    cstone::computeFixedGroups(firstBody, lastBody, bhMaxTargetSize(), groups);
    thrust::device_vector<int> globalPool(stackSize(groups.numGroups));

    double totalPotential = traverse(groups.view(), 1, x, y, z, m, h, x, y, z, m, h, childOffsets, internalToLeaf,
                                     layout, sourceCenter, Multipole, G, numShells, {box.lx(), box.ly(), box.lz()}, p,
                                     ax, ay, az, thrust::raw_pointer_cast(globalPool.data()));
    kernelSuccess("traverse");

    auto stats = readBhStats();
    return {Tc(stats[0]) / numBodies, Tc(stats[1]), Tc(stats[2]) / numBodies, Tc(stats[3]), Tc(totalPotential)};
}

template<class KeyType, class T, class MType>
void upsweep(int numSources, int numLeaves, int numLevels, float theta, const TreeNodeIndex* levelRange, const T* x,
             const T* y, const T* z, const T* m, const cstone::Box<T>& box, const LocalIndex* layout,
             const KeyType* prefixes, const TreeNodeIndex* childOffsets, const TreeNodeIndex* leafToInternal,
             Vec4<T>* centers, MType* Multipole)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    cstone::computeLeafSourceCenterGpu(x, y, z, m, leafToInternal, numLeaves, layout, centers);
    cstone::upsweepCentersGpu(cstone::maxTreeLevel<KeyType>{}, levelRange, childOffsets, centers);

    computeLeafMultipoles(x, y, z, m, leafToInternal, numLeaves, layout, centers, Multipole);
    for (int level = numLevels - 1; level >= 1; level--)
    {
        upsweepMultipoles(levelRange[level], levelRange[level + 1], childOffsets, centers, Multipole);
    }

    cstone::setMacGpu(prefixes, numSources, centers, 1.f / theta, box);

    auto   t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();

    fprintf(stdout, "Upward pass          : %.7f s\n", dt);
}
