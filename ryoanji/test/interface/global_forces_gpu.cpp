/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Compute an octree, multipoles and forces on GPUs from a set of particles distributed across ranks
 *        and compare against a single-node reference computed from the same set.
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

#include "ryoanji/interface/global_multipole.hpp"
#include "ryoanji/interface/multipole_holder.cuh"

using namespace ryoanji;

template<class T, class KeyType>
static int multipoleHolderTest(int thisRank, int numRanks)
{
    using MultipoleType              = CartesianQuadrupole<T>;
    const LocalIndex numParticles    = (100000 / numRanks) * numRanks;
    unsigned         bucketSize      = numParticles / (100 * numRanks);
    unsigned         bucketSizeLocal = std::min(64u, bucketSize);
    float            theta           = 0.5;
    T                G               = 1.0;

    cstone::Box<T> box{-1, 1};
    // setting numShells to non-zero won't match the direct sum reference because it only includes the replica shells,
    // but not the Ewald corrections, whereas the Barnes-Hut implementation contains both
    int numShells = 0;

    // common pool of coordinates, identical on all ranks
    cstone::RandomGaussianCoordinates<T, cstone::SfcKind<KeyType>> coords(numParticles, box);

    std::vector<T> globalH(numParticles, 0.1);
    adjustSmoothingLength<KeyType>(globalH.size(), 5, 10, coords.x(), coords.y(), coords.z(), globalH, box);

    std::vector<T> globalMasses(numParticles, 1.0 / numParticles);

    LocalIndex firstIndex = (numParticles * thisRank) / numRanks;
    LocalIndex lastIndex  = (numParticles * (thisRank + 1)) / numRanks;

    // extract a slice of the common pool, each rank takes a different slice, but all slices together
    // are equal to the common pool
    std::vector<T>       x(coords.x().begin() + firstIndex, coords.x().begin() + lastIndex);
    std::vector<T>       y(coords.y().begin() + firstIndex, coords.y().begin() + lastIndex);
    std::vector<T>       z(coords.z().begin() + firstIndex, coords.z().begin() + lastIndex);
    std::vector<T>       h(globalH.begin() + firstIndex, globalH.begin() + lastIndex);
    std::vector<T>       m(globalMasses.begin() + firstIndex, globalMasses.begin() + lastIndex);
    std::vector<KeyType> h_keys(x.size());

    cstone::Domain<KeyType, T, cstone::GpuTag> domain(thisRank, numRanks, bucketSize, bucketSizeLocal, theta, box);

    MultipoleHolder<T, T, T, T, T, KeyType, MultipoleType> multipoleHolder;

    cstone::DeviceVector<KeyType> d_keys = h_keys;
    cstone::DeviceVector<T>       d_x = x, d_y = y, d_z = z, d_h = h;
    cstone::DeviceVector<T>       d_m = m;
    cstone::DeviceVector<T>       s1;
    cstone::DeviceVector<T>       s2, s3;
    domain.syncGrav(d_keys, d_x, d_y, d_z, d_h, d_m, std::tuple{}, std::tie(s1, s2, s3));
    domain.exchangeHalos(std::tie(d_m), s1, s2);

    h_keys.resize(domain.nParticles());
    memcpyD2H(d_keys.data() + domain.startIndex(), domain.nParticles(), h_keys.data());

    /*! The range [firstGlobalIdx:lastGlobalIdx] of the global set @a coords is identical to the locally
     *  present particles contained in the range [domain.startIndex():domain.endIndex()] of arrays (d_x, d_y, d_z)
     *  This is true because @a coords is SFC sorted, therefore after domain.syncGrav, each rank will have
     *  one continuous section of the global @a coords set, which will largely overlap with the initial pre-sync range
     *  [firstIndex:lastIndex] but might be shifted left or right by a small distance.
     */
    LocalIndex firstGlobalIdx =
        std::lower_bound(coords.particleKeys().begin(), coords.particleKeys().end(), h_keys.front()) -
        coords.particleKeys().begin();
    LocalIndex lastGlobalIdx =
        std::upper_bound(coords.particleKeys().begin(), coords.particleKeys().end(), h_keys.back()) -
        coords.particleKeys().begin();

    //! includes tree plus associated information, like peer ranks, assignment, counts, centers, etc
    const cstone::FocusedOctree<KeyType, T, cstone::GpuTag>& focusTree = domain.focusTree();
    //! the focused octree, structure only
    auto                                         octree  = focusTree.octreeViewAcc();
    std::span<const cstone::SourceCenterType<T>> centers = focusTree.expansionCentersAcc();

    std::vector<MultipoleType> multipoles(octree.numNodes);
    multipoleHolder.upsweep(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), domain.globalTree(), domain.focusTree(),
                            domain.layout().data(), multipoles.data());

    // Check Barnes-Hut accelerations from distributed particle set
    // against the direct-sum reference computed with the (single node) common pool
    bool                pass;
    std::vector<double> firstPercentiles(numRanks), maxErrors(numRanks);
    {
        cstone::DeviceVector<T> d_ax, d_ay, d_az;
        d_ax = std::vector<T>(domain.nParticlesWithHalos(), 0);
        d_ay = std::vector<T>(domain.nParticlesWithHalos(), 0);
        d_az = std::vector<T>(domain.nParticlesWithHalos(), 0);

        auto   grp = multipoleHolder.computeSpatialGroups(domain.startIndex(), domain.endIndex(), rawPtr(d_x),
                                                          rawPtr(d_y), rawPtr(d_z), rawPtr(d_h), domain.focusTree(),
                                                          domain.layout().data(), domain.box());
        double bhPotential =
            multipoleHolder.compute(grp, rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), rawPtr(d_h), G, 0, box,
                                    nullptr, rawPtr(d_ax), rawPtr(d_ay), rawPtr(d_az));

        // create a host vector and download a device pointer range into it
        auto dl = [](auto* p1, auto* p2)
        {
            std::vector<std::remove_pointer_t<decltype(p1)>> ret(p2 - p1);
            memcpyD2H(p1, p2 - p1, ret.data());
            return ret;
        };

        // download BH accelerations for locally present particles
        std::vector<T> ax = dl(d_ax.data() + domain.startIndex(), d_ax.data() + domain.endIndex());
        std::vector<T> ay = dl(d_ay.data() + domain.startIndex(), d_ay.data() + domain.endIndex());
        std::vector<T> az = dl(d_az.data() + domain.startIndex(), d_az.data() + domain.endIndex());

        cstone::DeviceVector<T> d_xref = coords.x(), d_yref = coords.y(), d_zref = coords.z();
        cstone::DeviceVector<T> d_mref = globalMasses;
        cstone::DeviceVector<T> d_href = globalH;

        cstone::DeviceVector<T> d_potref(numParticles, 0);
        cstone::DeviceVector<T> d_axref(numParticles, 0), d_ayref(numParticles, 0), d_azref(numParticles, 0);

        // reference direct sum calculation with the global set of sources
        directSum(firstGlobalIdx, lastGlobalIdx, numParticles, {box.lx(), box.ly(), box.lz()}, numShells,
                  rawPtr(d_xref), rawPtr(d_yref), rawPtr(d_zref), rawPtr(d_mref), rawPtr(d_href), rawPtr(d_potref),
                  rawPtr(d_axref), rawPtr(d_ayref), rawPtr(d_azref));

        // download reference direct sum accelerations due to global source set
        std::vector<T> pRef  = dl(d_potref.data() + firstGlobalIdx, d_potref.data() + lastGlobalIdx);
        std::vector<T> axRef = dl(d_axref.data() + firstGlobalIdx, d_axref.data() + lastGlobalIdx);
        std::vector<T> ayRef = dl(d_ayref.data() + firstGlobalIdx, d_ayref.data() + lastGlobalIdx);
        std::vector<T> azRef = dl(d_azref.data() + firstGlobalIdx, d_azref.data() + lastGlobalIdx);

        double         potentialSumRef = 0;
        std::vector<T> errors(ax.size());
        for (int i = 0; i < ax.size(); i++)
        {
            potentialSumRef += pRef[i];
            Vec3<T> ref   = {axRef[i], ayRef[i], azRef[i]};
            Vec3<T> probe = {ax[i], ay[i], az[i]};
            errors[i]     = std::sqrt(norm2(ref - probe) / norm2(ref));
        }
        potentialSumRef *= 0.5 * G;
        std::sort(begin(errors), end(errors));

        double err1pc = errors[errors.size() * 0.99];
        double errmax = errors.back();

        MPI_Allgather(&err1pc, 1, MPI_DOUBLE, firstPercentiles.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(&errmax, 1, MPI_DOUBLE, maxErrors.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);

        double bhPotentialGlob, potentialSumRefGlob;
        MPI_Allreduce(&bhPotential, &bhPotentialGlob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&potentialSumRef, &potentialSumRefGlob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double atol1pc = 1e-3;
        double atolmax = 3e-2;
        double ptol    = 1e-2;

        if (thisRank == 0)
        {
            for (int i = 0; i < numRanks; ++i)
                std::cout << "rank " << i << " 1st-percentile acc error " << firstPercentiles[i] << ", max acc error "
                          << maxErrors[i] << std::endl;

            std::cout << "global reference potential " << potentialSumRefGlob << ", BH global potential "
                      << bhPotentialGlob << std::endl;
        }

        bool passAcc1pc = *std::max_element(firstPercentiles.begin(), firstPercentiles.end()) < atol1pc;
        bool passAccMax = *std::max_element(maxErrors.begin(), maxErrors.end()) < atolmax;
        bool passPot    = std::abs((bhPotentialGlob - potentialSumRefGlob) / potentialSumRefGlob) < ptol;

        pass = passAcc1pc && passAccMax && passPot;

        // Test separate targets and sources
        {
            auto extract = [](cstone::DeviceVector<T>& dv, LocalIndex a, LocalIndex b)
            {
                cstone::DeviceVector<T> ret(b - a);
                memcpyD2D(dv.data() + a, ret.size(), ret.data());
                return ret;
            };

            /* Compute accelerations for some target particles. We pick the first numTarget sources because
             * we already have the reference values. But the target particles can be distinct from the sources.
             * The only concern is performance: this only runs efficiently if the targets are SFC sorted and grouped
             * by MultipoleHolder::computeSpatialGroups to ensure that target groups have compact bounding boxes.
             */
            LocalIndex              numTargets = 100;
            cstone::DeviceVector<T> d_xt       = extract(d_x, domain.startIndex(), domain.startIndex() + numTargets);
            cstone::DeviceVector<T> d_yt       = extract(d_y, domain.startIndex(), domain.startIndex() + numTargets);
            cstone::DeviceVector<T> d_zt       = extract(d_z, domain.startIndex(), domain.startIndex() + numTargets);
            cstone::DeviceVector<T> d_mt       = extract(d_m, domain.startIndex(), domain.startIndex() + numTargets);
            cstone::DeviceVector<T> d_ht       = extract(d_h, domain.startIndex(), domain.startIndex() + numTargets);

            auto targetGroups =
                multipoleHolder.computeSpatialGroups(0, numTargets, d_xt.data(), d_yt.data(), d_zt.data(), d_ht.data(),
                                                     focusTree, domain.layout().data(), box);

            cstone::DeviceVector<T> d_axt(numTargets, 0), d_ayt(numTargets, 0), d_azt(numTargets, 0);

            multipoleHolder.compute(targetGroups, d_xt.data(), d_yt.data(), d_zt.data(), d_mt.data(), d_ht.data(),
                                    d_x.data(), d_y.data(), d_z.data(), d_m.data(), d_h.data(), G, 0, box, nullptr,
                                    rawPtr(d_axt), rawPtr(d_ayt), rawPtr(d_azt));

            std::vector<T> axt = toHost(d_axt);
            std::vector<T> ayt = toHost(d_ayt);
            std::vector<T> azt = toHost(d_azt);

            bool passTargets = true;
            for (int i = 0; i < numTargets; i++)
            {
                Vec3<T> ref   = {axRef[i], ayRef[i], azRef[i]};
                Vec3<T> probe = {axt[i], ayt[i], azt[i]};
                T       error = std::sqrt(norm2(ref - probe) / norm2(ref));
                if (error > atolmax) { passTargets = false; }
            }
            if (not passTargets) { std::cout << "Separate targets were not correct" << std::endl; }
            pass = pass && passTargets;
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
    MPI_Init(NULL, NULL);

    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int testResult = multipoleHolderTest<float, uint64_t>(rank, numRanks);

    MPI_Finalize();

    return testResult;
}
