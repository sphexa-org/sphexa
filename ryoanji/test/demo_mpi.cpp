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
 * @brief GTest MPI driver
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <chrono>
#include <mpi.h>

#define USE_CUDA
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/domain/domain.hpp"
#include "coord_samples/random.hpp"

#include "ryoanji/interface/multipole_holder.cuh"
#include "ryoanji/nbody/cartesian_qpole.hpp"

using AccType = cstone::GpuTag;

using namespace ryoanji;

template<class T, class KeyType>
void ryoanjiTest(int thisRank, int numRanks, size_t numParticlesGlobal)
{
    constexpr int P                = 4;
    using MultipoleType            = SphericalMultipole<T, P>;
    size_t   numParticles          = numParticlesGlobal / numRanks;
    unsigned bucketSizeFocus       = 64;
    unsigned numGlobalNodesPerRank = 100;
    unsigned bucketSizeGlobal =
        std::max(size_t(bucketSizeFocus), numParticlesGlobal / (numGlobalNodesPerRank * numRanks));
    T              G     = 1.0;
    float          theta = 0.5;
    cstone::Box<T> box{-1, 1};

    // radius that includes ~30 neighbor particles on average
    double hmean = std::cbrt(30.0 / 0.523 / double(numParticlesGlobal) * box.lx() * box.ly() * box.lz());

    int                                                    seed = thisRank;
    cstone::RandomCoordinates<T, cstone::SfcKind<KeyType>> coords(numParticles, box, seed);

    std::vector<T> h(numParticles, hmean);
    std::vector<T> m(numParticles, 1.0f / float(numParticlesGlobal));

    cstone::Domain<KeyType, T, AccType> domain(thisRank, numRanks, bucketSizeGlobal, bucketSizeFocus, theta, box);

    // upload particles to GPU
    cstone::DeviceVector<KeyType> d_keys = std::vector<KeyType>(numParticles);
    cstone::DeviceVector<T>       d_x    = coords.x();
    cstone::DeviceVector<T>       d_y    = coords.y();
    cstone::DeviceVector<T>       d_z    = coords.z();
    cstone::DeviceVector<T>       d_h    = h;
    cstone::DeviceVector<T>       d_m    = m;
    cstone::DeviceVector<T>       s1, s2, s3; // scratch buffers for sorting, reordering, etc

    //! Build octrees, decompose domain and distribute particles and halos, may resize buffers
    domain.syncGrav(d_keys, d_x, d_y, d_z, d_h, d_m, std::tuple{}, std::tie(s1, s2, s3));
    //! halos for x,y,z,h are already exchanged in syncGrav
    domain.exchangeHalos(std::tie(d_m), s1, s2);

    cstone::DeviceVector<T> d_ax = std::vector<T>(domain.nParticlesWithHalos());
    cstone::DeviceVector<T> d_ay = std::vector<T>(domain.nParticlesWithHalos());
    cstone::DeviceVector<T> d_az = std::vector<T>(domain.nParticlesWithHalos());

    //! includes tree plus associated information, like nearby ranks, assignment, counts, MAC spheres, etc
    const cstone::FocusedOctree<KeyType, T, cstone::GpuTag>& focusTree = domain.focusTree();
    //! the focused octree on GPU, structure only
    auto octree = focusTree.octreeViewAcc();

    MultipoleHolder<T, T, T, T, T, KeyType, MultipoleType> multipoleHolder;

    std::vector<MultipoleType> multipoles(octree.numNodes);
    auto grp = multipoleHolder.computeSpatialGroups(domain.startIndex(), domain.endIndex(), rawPtr(d_x), rawPtr(d_y),
                                                    rawPtr(d_z), rawPtr(d_h), domain.focusTree(),
                                                    domain.layout().data(), domain.box());
    multipoleHolder.upsweep(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), domain.globalTree(), domain.focusTree(),
                            domain.layout().data(), multipoles.data());

    auto t0 = std::chrono::high_resolution_clock::now();
    // compute accelerations for locally owned particles based on globally valid multipoles and
    // halo particles are in [0:domain.startIndex()] and in [domain.endIndex():domain.nParticlesWithHalos()]
    float totalPotential = multipoleHolder.compute(grp, rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), rawPtr(d_h),
                                                   G, 0, box, nullptr, rawPtr(d_ax), rawPtr(d_ay), rawPtr(d_az));

    auto t1 = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration<double>(t1 - t0).count();

    float totalPotentialGlobal;
    mpiAllreduce(&totalPotential, &totalPotentialGlobal, 1, MPI_SUM);

    auto [numP2P, maxP2P, numM2P, maxM2P, maxStack] = multipoleHolder.readStats();
    double flops                                    = (numP2P * 23 + numM2P * 2 * pow(P, 3)) / dt / 1e12;

    for (int rank = 0; rank < numRanks; ++rank)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == thisRank)
        {
            if (thisRank == 0) { fprintf(stdout, "Total potential          : %.7f\n", totalPotentialGlobal); }
            fprintf(stdout, "--- rank %d ----------------\n", thisRank);
            fprintf(stdout, "numParticles, numHalos   : %d, %d\n", domain.nParticles(),
                    domain.nParticlesWithHalos() - domain.nParticles());
            fprintf(stdout, "BH                       : %.7f s (%.7f TFlops)\n", dt, flops);
        }
    }
}

int main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);

    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t numParticles = argc > 1 ? std::stoll(argv[1]) : 1000000ll;

    ryoanjiTest<float, uint64_t>(rank, numRanks, numParticles);

    MPI_Finalize();
}
