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
 * @brief Compute an octree and multipoles on GPUs from a set of particles distributed across ranks
 *        and compare against a single-node reference computed from the same set.
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <mpi.h>

#define USE_CUDA
#include "cstone/cuda/cuda_utils.cuh"
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
    const LocalIndex numParticles    = 1000 * numRanks;
    unsigned         bucketSize      = 64;
    unsigned         bucketSizeLocal = 16;
    float            theta           = 10.0;

    cstone::Box<T> box{-1, 1};

    // common pool of coordinates, identical on all ranks
    cstone::RandomGaussianCoordinates<T, cstone::SfcKind<KeyType>> coords(numRanks * numParticles, box);

    std::vector<T> globalH(numRanks * numParticles, 0.1);
    adjustSmoothingLength<KeyType>(globalH.size(), 5, 10, coords.x(), coords.y(), coords.z(), globalH, box);

    std::vector<T> globalMasses(numRanks * numParticles, 1.0 / (numRanks * numParticles));

    auto firstIndex = numParticles * thisRank;
    auto lastIndex  = numParticles * thisRank + numParticles;

    // extract a slice of the common pool, each rank takes a different slice, but all slices together
    // are equal to the common pool
    std::vector<T> x(coords.x().begin() + firstIndex, coords.x().begin() + lastIndex);
    std::vector<T> y(coords.y().begin() + firstIndex, coords.y().begin() + lastIndex);
    std::vector<T> z(coords.z().begin() + firstIndex, coords.z().begin() + lastIndex);
    std::vector<T> h(globalH.begin() + firstIndex, globalH.begin() + lastIndex);
    std::vector<T> m(globalMasses.begin() + firstIndex, globalMasses.begin() + lastIndex);

    std::vector<KeyType> particleKeys(x.size());

    cstone::Domain<KeyType, T, cstone::GpuTag> domain(thisRank, numRanks, bucketSize, bucketSizeLocal, theta, box);

    MultipoleHolder<T, T, T, T, T, KeyType, MultipoleType> multipoleHolder;

    cstone::DeviceVector<KeyType> d_keys = particleKeys;
    cstone::DeviceVector<T>       d_x = x, d_y = y, d_z = z, d_h = h, d_m = m;
    cstone::DeviceVector<T>       s1, s2, s3;
    domain.syncGrav(d_keys, d_x, d_y, d_z, d_h, d_m, std::tuple{}, std::tie(s1, s2, s3));
    domain.exchangeHalos(std::tie(d_m), s1, s2);

    //! includes tree plus associated information, like peer ranks, assignment, counts, centers, etc
    const cstone::FocusedOctree<KeyType, T, cstone::GpuTag>& focusTree = domain.focusTree();
    //! the focused octree, structure only
    auto octree = focusTree.octreeViewAcc();

    std::vector<MultipoleType> multipoles(octree.numNodes);
    multipoleHolder.upsweep(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), domain.globalTree(), domain.focusTree(),
                            domain.layout().data(), multipoles.data());

    // Check the root multipole of the distributed tree
    bool passMultipole = false;
    {
        memcpyD2H(multipoleHolder.deviceMultipoles(), multipoles.size(), multipoles.data());

        MultipoleType globalRootMultipole = multipoles[0];

        auto                                     d_centers = focusTree.expansionCentersAcc();
        std::vector<cstone::SourceCenterType<T>> centers(d_centers.size());
        memcpyD2H(d_centers.data(), d_centers.size(), centers.data());

        // compute reference root cell multipole from global particle data
        MultipoleType reference;
        P2M(coords.x().data(), coords.y().data(), coords.z().data(), globalMasses.data(), 0, numParticles * numRanks,
            centers[0], reference);

        double maxDiff = max(abs(reference - globalRootMultipole));

        bool pass      = maxDiff < 1e-10;
        int  numPassed = pass;
        mpiAllreduce(MPI_IN_PLACE, &numPassed, 1, MPI_SUM);
        if (numPassed == numRanks) { passMultipole = true; }
    }

    if (thisRank == 0)
    {
        std::string testResult = passMultipole ? "PASS" : "FAIL";
        std::cout << "Upsweep test result: " << testResult << std::endl;
    }

    if (passMultipole) { return EXIT_SUCCESS; }
    else { return EXIT_FAILURE; }
}

int main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);

    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int testResult = multipoleHolderTest<double, uint64_t>(rank, numRanks);

    MPI_Finalize();

    return testResult;
}
