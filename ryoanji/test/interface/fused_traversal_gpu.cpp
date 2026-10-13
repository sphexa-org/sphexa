/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Compare the fused in-register group-splitting gravity traversal against the classic
 *        path with precomputed groups, and the CPU leaf-range traversal, from identical state.
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <mpi.h>

#define USE_CUDA
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/device_vector.h"
#include "cstone/domain/domain.hpp"
#include "cstone/findneighbors.hpp"
#include "coord_samples/random.hpp"

#include "ryoanji/interface/multipole_holder.cuh"
#include "ryoanji/nbody/traversal_cpu.hpp"
#include "ryoanji/nbody/traversal_gpu.h"

using namespace ryoanji;

/*! @brief Verify that the group-free fused traversal reproduces the classic group-based path
 *
 * One domain and particle set is built, the multipoles are computed once, then
 * 1. the classic path (computeSpatialGroups + group-based compute) and
 * 2. the fused path (group-free compute with in-register group splitting)
 * are run from identical starting state. Per-particle accelerations ax/ay/az and per-particle
 * potentials ugrav must be bit-identical between the two paths, only the total potential
 * (accumulated with atomicAdd in nondeterministic order) is compared within a tolerance.
 * The CPU leaf-range traversal is additionally computed from the same state and compared
 * to the fused GPU result within the Barnes-Hut approximation tolerance.
 */
template<class T, class KeyType>
static int fusedTraversalTest(int thisRank, int numRanks)
{
    using MultipoleType           = CartesianQuadrupole<T>;
    const LocalIndex numParticles = (100000 / numRanks) * numRanks;
    unsigned         bucketSize   = numParticles / (100 * numRanks);
    unsigned         bucketSizeLocal = std::min(64u, bucketSize);
    float            theta           = 0.5;
    T                G               = 1.0;

    // Fixed boundary to prevent the domain from computing a tightly fitting box, because domain.sync would then
    // change the relative SFC order of the local particles w.r.t to the global reference set.
    cstone::Box<T> box(-1, 1, cstone::BoundaryType::fixed);
    // setting numShells to non-zero won't match the CPU reference because it only includes the replica shells,
    // but not the Ewald corrections, whereas the Barnes-Hut implementation contains both
    int numShells = 0;

    // common pool of coordinates, identical on all ranks
    cstone::RandomGaussianCoordinates<T, cstone::SfcKind<KeyType>> coords(numParticles, box);
    coords.adjustH(5, 10);

    std::vector<T> globalMasses(numParticles, 1.0 / numParticles);

    LocalIndex firstIndex = (numParticles * thisRank) / numRanks;
    LocalIndex lastIndex  = (numParticles * (thisRank + 1)) / numRanks;

    // extract a slice of the common pool, each rank takes a different slice, but all slices together
    // are equal to the common pool
    std::vector<T>       x(coords.x().begin() + firstIndex, coords.x().begin() + lastIndex);
    std::vector<T>       y(coords.y().begin() + firstIndex, coords.y().begin() + lastIndex);
    std::vector<T>       z(coords.z().begin() + firstIndex, coords.z().begin() + lastIndex);
    std::vector<T>       h(coords.h().begin() + firstIndex, coords.h().begin() + lastIndex);
    std::vector<T>       m(globalMasses.begin() + firstIndex, globalMasses.begin() + lastIndex);
    std::vector<KeyType> h_keys(x.size());

    cstone::Domain<KeyType, T, cstone::execution::Gpu> domain(cstone::execution::gpuDefaultStream, thisRank, numRanks,
                                                              bucketSize, bucketSizeLocal, theta, MPI_COMM_WORLD, box);

    MultipoleHolder<T, T, T, T, T, KeyType, MultipoleType> multipoleHolder;

    cstone::DeviceVector<KeyType> d_keys = h_keys;
    cstone::DeviceVector<T>       d_x = x, d_y = y, d_z = z, d_h = h;
    cstone::DeviceVector<T>       d_m = m;
    cstone::DeviceVector<T>       s1;
    cstone::DeviceVector<T>       s2, s3;
    domain.syncGrav(d_keys, d_x, d_y, d_z, d_h, d_m, std::tuple{}, std::tie(s1, s2, s3));
    domain.exchangeHalos(std::tie(d_m), s1, s2);

    multipoleHolder.upsweep(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), domain.globalTree(), domain.focusTree(),
                            domain.layout().data());

    //! includes tree plus associated information, like peer ranks, assignment, counts, centers, etc
    const cstone::FocusedOctree<KeyType, T, cstone::execution::Gpu>& focusTree = domain.focusTree();
    //! the focused octree, structure only
    auto                                         octree  = focusTree.octreeViewAcc();
    std::span<const cstone::SourceCenterType<T>> centers = focusTree.expansionCentersAcc();

    bool pass = true;

    // classic path with precomputed groups
    cstone::DeviceVector<T> d_ax1(domain.nParticlesWithHalos(), 0), d_ay1(domain.nParticlesWithHalos(), 0);
    cstone::DeviceVector<T> d_az1(domain.nParticlesWithHalos(), 0), d_ugrav1(domain.nParticlesWithHalos(), 0);
    double                  egravClassic = 0;
    cstone::GroupView       grp;
    {
        grp = multipoleHolder.computeSpatialGroups(domain.startIndex(), domain.endIndex(), rawPtr(d_x), rawPtr(d_y),
                                                   rawPtr(d_z), rawPtr(d_h), domain.focusTree(),
                                                   domain.layout().data(), domain.box());
        egravClassic = multipoleHolder.compute(grp, rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), rawPtr(d_h), G,
                                               numShells, box, rawPtr(d_ugrav1), rawPtr(d_ax1), rawPtr(d_ay1),
                                               rawPtr(d_az1));
    }

    // fused path with in-register group splitting, from identical starting state
    cstone::DeviceVector<T> d_ax2(domain.nParticlesWithHalos(), 0), d_ay2(domain.nParticlesWithHalos(), 0);
    cstone::DeviceVector<T> d_az2(domain.nParticlesWithHalos(), 0), d_ugrav2(domain.nParticlesWithHalos(), 0);
    double                  egravFused = 0;
    {
        egravFused = multipoleHolder.compute(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), rawPtr(d_h), G,
                                             numShells, domain.startIndex(), domain.endIndex(), domain.focusTree(),
                                             domain.layout().data(), domain.box(), rawPtr(d_ugrav2), rawPtr(d_ax2),
                                             rawPtr(d_ay2), rawPtr(d_az2));
    }

    // per-particle results must be bit-identical between the two paths
    {
        auto download = [thisRank](const cstone::DeviceVector<T>& dv)
        {
            std::vector<T> ret(dv.size());
            cstone::memcpyD2HAsync(cstone::execution::gpuDefaultStream, dv.data(), dv.size(), ret.data());
            cstone::syncGpu(cstone::execution::gpuDefaultStream);
            return ret;
        };

        std::vector<T> ax1 = download(d_ax1), ay1 = download(d_ay1), az1 = download(d_az1), ugrav1 = download(d_ugrav1);
        std::vector<T> ax2 = download(d_ax2), ay2 = download(d_ay2), az2 = download(d_az2), ugrav2 = download(d_ugrav2);

        LocalIndex numDiff    = 0;
        LocalIndex numFixedGroups =
            cstone::iceil(domain.endIndex() - domain.startIndex(), LocalIndex(bhMaxTargetSize()));
        for (LocalIndex i = domain.startIndex(); i < domain.endIndex(); ++i)
        {
            numDiff += int(ax1[i] != ax2[i]) + int(ay1[i] != ay2[i]) + int(az1[i] != az2[i]) + int(ugrav1[i] != ugrav2[i]);
        }

        if (thisRank == 0)
        {
            std::cout << "classic groups: " << grp.numGroups << ", fixed groups: " << numFixedGroups
                      << ", differing per-particle results: " << numDiff << std::endl;
        }
        if (numDiff != 0)
        {
            std::cout << "Rank " << thisRank << ": fused traversal is not bit-identical to the classic path"
                      << std::endl;
            pass = false;
        }

        // total potential is accumulated with atomicAdd in nondeterministic order, compare within tolerance
        double egravClassicGlob, egravFusedGlob;
        MPI_Allreduce(&egravClassic, &egravClassicGlob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&egravFused, &egravFusedGlob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double potTol = 1e-4;
        if (std::abs((egravFusedGlob - egravClassicGlob) / egravClassicGlob) > potTol)
        {
            if (thisRank == 0)
            {
                std::cout << "total potential mismatch: classic " << egravClassicGlob << ", fused " << egravFusedGlob
                          << std::endl;
            }
            pass = false;
        }
    }

    // CPU leaf-range traversal from identical state, compared to the fused GPU result
    // within the Barnes-Hut approximation tolerance
    {
        auto cpToHost = []<class X>(const X* ptr, int n)
        {
            std::vector<X> ret(n);
            cstone::memcpyD2HAsync(cstone::execution::gpuDefaultStream, ptr, n, ret.data());
            cstone::syncGpu(cstone::execution::gpuDefaultStream);
            return ret;
        };

        auto child          = cpToHost(octree.childOffsets, octree.numNodes);
        auto parents        = cpToHost(octree.parents, octree.numParents);
        auto intToLeaf      = cpToHost(octree.internalToLeaf, octree.numNodes);
        auto centers_cpu    = cpToHost(centers.data(), centers.size());
        auto multipoles_cpu = cpToHost(multipoleHolder.deviceMultipoles(), octree.numNodes);
        auto layout         = cpToHost(domain.layout().data(), domain.layout().size());

        auto x_cpu = cpToHost(d_x.data(), int(d_x.size()));
        auto y_cpu = cpToHost(d_y.data(), int(d_y.size()));
        auto z_cpu = cpToHost(d_z.data(), int(d_z.size()));
        auto h_cpu = cpToHost(d_h.data(), int(d_h.size()));
        auto m_cpu = cpToHost(d_m.data(), int(d_m.size()));

        std::vector<T> ax_cpu(x_cpu.size(), 0), ay_cpu(x_cpu.size(), 0), az_cpu(x_cpu.size(), 0);

        T egravTot_cpu = 0;
        computeGravity(child.data(), parents.data(), intToLeaf.data(), centers_cpu.data(), multipoles_cpu.data(),
                       layout.data(), 0, octree.numLeafNodes, x_cpu.data(), y_cpu.data(), z_cpu.data(), h_cpu.data(),
                       m_cpu.data(), box, G, (T*)nullptr, ax_cpu.data(), ay_cpu.data(), az_cpu.data(), &egravTot_cpu,
                       numShells);

        auto dl = [](auto* p1, auto* p2)
        {
            std::vector<std::remove_pointer_t<decltype(p1)>> ret(p2 - p1);
            cstone::memcpyD2HAsync(cstone::execution::gpuDefaultStream, p1, p2 - p1, ret.data());
            cstone::syncGpu(cstone::execution::gpuDefaultStream);
            return ret;
        };

        // download fused GPU accelerations for locally present particles
        std::vector<T> ax = dl(d_ax2.data() + domain.startIndex(), d_ax2.data() + domain.endIndex());
        std::vector<T> ay = dl(d_ay2.data() + domain.startIndex(), d_ay2.data() + domain.endIndex());
        std::vector<T> az = dl(d_az2.data() + domain.startIndex(), d_az2.data() + domain.endIndex());

        double maxRelError = 0;
        for (std::size_t i = 0; i < ax.size(); i++)
        {
            Vec3<T> cpu{ax_cpu[i + domain.startIndex()], ay_cpu[i + domain.startIndex()], az_cpu[i + domain.startIndex()]};
            Vec3<T> gpu{ax[i], ay[i], az[i]};
            T       error = std::sqrt(norm2(cpu - gpu) / norm2(cpu));
            maxRelError    = std::max(maxRelError, double(error));
        }

        double maxRelErrorGlob;
        MPI_Allreduce(&maxRelError, &maxRelErrorGlob, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (thisRank == 0)
        {
            std::cout << "CPU leaf-range vs fused GPU max relative acc error " << maxRelErrorGlob << std::endl;
        }
        // CPU and GPU Barnes-Hut traversals visit a slightly different set of tree cells,
        // their mutual deviation is bounded by the Barnes-Hut approximation error
        double accTol = 3e-2;
        if (maxRelErrorGlob > accTol)
        {
            std::cout << "Rank " << thisRank << ": CPU traversal disagrees with fused GPU traversal" << std::endl;
            pass = false;
        }
    }

    if (thisRank == 0)
    {
        std::string testResult = pass ? "PASS" : "FAIL";
        std::cout << "Test result: " << testResult << std::endl;
    }

    if (pass) { return EXIT_SUCCESS; }
    else { return EXIT_FAILURE; }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int testResult = fusedTraversalTest<float, uint64_t>(rank, numRanks);
    testResult     = std::max(testResult, fusedTraversalTest<double, uint64_t>(rank, numRanks));

    MPI_Finalize();

    return testResult;
}
