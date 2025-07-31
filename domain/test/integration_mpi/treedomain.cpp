/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Test global octree build together with domain particle exchange
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <mpi.h>
#include <gtest/gtest.h>

#include "coord_samples/random.hpp"
#include "cstone/domain/layout.hpp"
#include "cstone/domain/domaindecomp_mpi.hpp"
#include "cstone/tree/update_mpi.hpp"
#include "cstone/tree/cs_util.hpp"
#include "cstone/util/reallocate.hpp"

using namespace cstone;

/*! @brief integration test between global octree build and domain particle exchange
 *
 * This test performs the following steps on each rank:
 *
 * 1. Create numParticles randomly
 *
 *    RandomGaussianCoordinates<T, KeyType> coords(numParticles, box, thisRank):
 *    Creates nParticles in the [-1,1]^3 box with random gaussian distribution
 *    centered at (0,0,0), calculate the Morton code for each particle,
 *    reorder codes and x,y,z array of coords according to ascending Morton codes
 *
 * 2. Compute global octree and node particle counts
 *
 *    auto [tree, counts] = computeOctreeGlobal(...)
 *
 * 3. split & assign a part of the global octree to each rank
 *
 * 4. Exchange particles, such that each rank gets all the particles present on all nodes
 *    that lie within the area of the octree that each rank got assigned.
 *
 * Post exchange:
 *
 * 5. Repeat 2., global octree and counts should stay the same
 *
 * 6. Repeat 3., the decomposition should now indicate that all particles stay on the
 *    node they currently are and that no rank sends any particles to other ranks.
 */
template<class KeyType, class T, template<class> class sfcKeyType>
void globalRandomGaussian(int thisRank, int numRanks, const Box<T>& box)
{
    LocalIndex numParticles = 1000;
    unsigned bucketSize     = 64;

    const auto mixDBits = getBoxMixDimensionBits<T, KeyType, Box<T>>(box);
    const bool useMixD = (mixDBits.bx != maxTreeLevel<KeyType>{} ||
                          mixDBits.by != maxTreeLevel<KeyType>{} ||
                          mixDBits.bz != maxTreeLevel<KeyType>{});

    RandomCoordinates<T, sfcKeyType<KeyType>> coords =
        useMixD
            ? RandomCoordinates<T, sfcKeyType<KeyType>>{numParticles, box, thisRank, mixDBits.bx, mixDBits.by, mixDBits.bz}
            : RandomCoordinates<T, sfcKeyType<KeyType>>{numParticles, box, thisRank};

    std::vector<KeyType> tree = makeRootNodeTree<KeyType>();
    std::vector<unsigned> counts{numRanks * unsigned(numParticles)};

    while (!updateOctreeGlobal<KeyType>(coords.particleKeys(), bucketSize, tree, counts)) {}

    std::vector<LocalIndex> ordering(numParticles);
    // particles are in SFC order
    std::iota(begin(ordering), end(ordering), 0);

    auto assignment = makeSfcAssignment(numRanks, counts, tree.data());
    auto sends      = createSendRanges<KeyType>(assignment, coords.particleKeys());

    EXPECT_EQ(std::accumulate(begin(counts), end(counts), std::size_t(0)), numParticles * numRanks);

    std::vector<T> x(coords.x().begin(), coords.x().end());
    std::vector<T> y(coords.y().begin(), coords.y().end());
    std::vector<T> z(coords.z().begin(), coords.z().end());

    LocalIndex numAssigned = assignment.totalCount(thisRank);
    LocalIndex numPresent  = sends.count(thisRank);

    BufferDescription bufDesc{0, numParticles, numParticles};
    bufDesc.size = domain_exchange::exchangeBufferSize(bufDesc, numPresent, numAssigned);
    reallocate(bufDesc.size, 1.01, x, y, z);
    ExchangeLog log;
    auto recvStart = domain_exchange::receiveStart(bufDesc, numAssigned - numPresent);
    auto recvEnd   = recvStart + numAssigned - numPresent;
    exchangeParticles(0, log, sends, thisRank, recvStart, recvEnd, ordering.data(), x.data(), y.data(), z.data());

    domain_exchange::extractLocallyOwned(bufDesc, numPresent, numAssigned, ordering.data() + sends[thisRank], x, y, z);

    /// post-exchange test:
    /// if the global tree build and assignment is repeated, no particles are exchanged anymore

    std::vector<KeyType> newCodes(x.size());
    if (useMixD)
    {
        computeSfcMixDKeys(x.data(), y.data(), z.data(), SfcMixDKindPointer(newCodes.data()), x.size(), box, mixDBits.bx,
                           mixDBits.by, mixDBits.bz);
    }
    else
    {
        computeSfcKeys(x.data(), y.data(), z.data(), sfcKindPointer(newCodes.data()), x.size(), box);
    }

    // received particles are not stored in SFC order after the exchange
    std::sort(begin(newCodes), end(newCodes));

    //! what particles we actually have should equal what was assigned
    EXPECT_EQ(numAssigned, x.size());

    std::vector<KeyType> newTree = makeRootNodeTree<KeyType>();
    std::vector<unsigned> newCounts{unsigned(x.size())};
    while (!updateOctreeGlobal<KeyType>(newCodes, bucketSize, newTree, newCounts))
        ;

    // global tree and counts stay the same
    EXPECT_EQ(tree, newTree);
    EXPECT_EQ(counts, newCounts);

    auto newSends = createSendRanges<KeyType>(assignment, {newCodes.data(), numAssigned});

    for (int rank = 0; rank < numRanks; ++rank)
    {
        // the new send list now indicates that all elements on the current rank
        // stay where they are
        if (rank == thisRank) EXPECT_EQ(newSends.count(rank), numAssigned);
        // no particles are sent to other ranks
        else
            EXPECT_EQ(newSends.count(rank), 0);
    }
}

TEST(GlobalTreeDomain, randomGaussian)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    globalRandomGaussian<uint64_t, double, SfcKind>(rank, nRanks, {-1, 1});
    globalRandomGaussian<unsigned, float, SfcKind>(rank, nRanks, {-1, 1});
    globalRandomGaussian<uint64_t, float, SfcKind>(rank, nRanks, {-1, 1});
    globalRandomGaussian<uint64_t, double, SfcMixDKind>(rank, nRanks, {0, 1, 0, 0.015625, 0, 0.00390625});
    globalRandomGaussian<unsigned, float, SfcMixDKind>(rank, nRanks, {0, 1, 0, 0.015625, 0, 0.00390625});
    globalRandomGaussian<uint64_t, float, SfcMixDKind>(rank, nRanks, {0, 1, 0, 0.015625, 0, 0.00390625});
}