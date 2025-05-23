/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Tests the particle exchange used for exchanging assigned particles, i.e. not halos.
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>

#include <gtest/gtest.h>

#include "cstone/domain/domaindecomp_mpi.hpp"
#include "cstone/util/reallocate.hpp"

using namespace cstone;
namespace ex = domain_exchange;

/*! @brief all-to-all exchange, the most communication possible
 *
 * @tparam T         float, double or int
 * @param thisRank   executing rank
 * @param numRanks   total number of ranks
 *
 * Each rank keeps (1/numRanks)-th of its local elements and sends the
 * other numRanks-1 chunks to the other numRanks-1 ranks.
 */
template<class T>
void exchangeAllToAll(int thisRank, int numRanks)
{
    LocalIndex gridSize = 64;

    std::vector<T> x(gridSize), y(gridSize);
    std::vector<LocalIndex> ordering(gridSize);

    std::iota(begin(x), end(x), 0);
    // unique element id across all ranks
    std::iota(begin(y), end(y), gridSize * thisRank);
    // start from trivial ordering
    std::iota(begin(ordering), end(ordering), 0);

    {
        // A simple, but nontrivial ordering.
        // Simulates the use case where the x,y,z coordinate arrays
        // are not sorted according to the SFC ordering for which
        // the index ranges in the SendList are valid.
        int swap1 = 0;
        int swap2 = gridSize - 1;
        std::swap(x[swap1], x[swap2]);
        std::swap(y[swap1], y[swap2]);
        std::swap(ordering[swap1], ordering[swap2]);
    }

    int segmentSize = gridSize / numRanks;

    SendRanges sends(numRanks + 1);
    sends.back() = gridSize;
    for (int rank = 0; rank < numRanks; ++rank)
    {
        int lower   = rank * segmentSize;
        sends[rank] = lower;
    }

    BufferDescription bufDesc{0, gridSize, gridSize};
    LocalIndex numPartPresent  = sends.count(thisRank);
    LocalIndex numPartAssigned = numPartPresent * numRanks;
    bufDesc.size               = ex::exchangeBufferSize(bufDesc, numPartPresent, numPartAssigned);
    reallocate(bufDesc.size, 1.01, x, y);

    ExchangeLog log;
    auto recvStart = domain_exchange::receiveStart(bufDesc, numPartAssigned - numPartPresent);
    auto recvEnd   = recvStart + numPartAssigned - numPartPresent;
    exchangeParticles(0, log, sends, thisRank, recvStart, recvEnd, ordering.data(), x.data(), y.data());

    ex::extractLocallyOwned(bufDesc, numPartPresent, numPartAssigned, ordering.data() + sends[thisRank], x, y);

    std::vector<T> refX(numPartAssigned);
    for (int rank = 0; rank < numRanks; ++rank)
    {
        std::iota(begin(refX) + rank * numPartPresent, begin(refX) + rank * numPartPresent + numPartPresent,
                  sends[thisRank]);
    }

    std::vector<T> refY;
    for (int rank = 0; rank < numRanks; ++rank)
    {
        int seqStart = rank * gridSize + (gridSize / numRanks) * thisRank;

        for (int i = 0; i < numPartPresent; ++i)
            refY.push_back(seqStart++);
    }

    std::sort(begin(x), end(x));
    std::sort(begin(refX), end(refX));

    // received particles are in indeterminate order
    std::sort(begin(y), end(y));

    EXPECT_EQ(refX, x);
    EXPECT_EQ(refY, y);
}

TEST(GlobalDomain, exchangeAllToAll)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    exchangeAllToAll<double>(rank, nRanks);
    MPI_Barrier(MPI_COMM_WORLD);
    exchangeAllToAll<float>(rank, nRanks);
    MPI_Barrier(MPI_COMM_WORLD);
    exchangeAllToAll<int>(rank, nRanks);
    MPI_Barrier(MPI_COMM_WORLD);
}

void exchangeCyclicNeighbors(int thisRank, int numRanks)
{
    LocalIndex gridSize = 64;

    // x and y are filled with one value that is different for each rank
    std::vector<double> x(gridSize, thisRank);
    std::vector<float> y(gridSize, -thisRank);
    std::vector<util::array<int, 2>> testArray(gridSize, {thisRank, -thisRank});
    std::vector<uint8_t> uint8Array(gridSize, thisRank);

    std::vector<LocalIndex> ordering(gridSize);
    std::iota(begin(ordering), end(ordering), 0);

    // send the last nex elements to the next rank
    int nex      = 10;
    int nextRank = (thisRank + 1) % numRanks;

    SendRanges sends(numRanks + 1);
    // keep all but the last nex elements
    sends[thisRank] = gridSize - nex;
    // send last nex to nextRank
    sends[nextRank] = nex;
    std::exclusive_scan(sends.begin(), sends.end(), sends.begin(), 0);

    BufferDescription bufDesc{0, gridSize, gridSize};
    LocalIndex numPartPresent  = sends.count(thisRank);
    LocalIndex numPartAssigned = gridSize;
    bufDesc.size               = ex::exchangeBufferSize(bufDesc, numPartPresent, numPartAssigned);
    reallocate(bufDesc.size, 1.01, x, y, testArray, uint8Array);

    ExchangeLog log;
    auto recvStart = domain_exchange::receiveStart(bufDesc, numPartAssigned - numPartPresent);
    auto recvEnd   = recvStart + numPartAssigned - numPartPresent;
    exchangeParticles(0, log, sends, thisRank, recvStart, recvEnd, ordering.data(), x.data(), y.data(),
                      uint8Array.data(), testArray.data());

    ex::extractLocallyOwned(bufDesc, numPartPresent, numPartAssigned, ordering.data() + sends[thisRank], x, y,
                            uint8Array, testArray);

    int incomingRank = (thisRank - 1 + numRanks) % numRanks;
    std::vector<double> refX(gridSize, thisRank);
    std::fill(begin(refX) + gridSize - nex, end(refX), incomingRank);

    std::vector<float> refY(gridSize, -thisRank);
    std::fill(begin(refY) + gridSize - nex, end(refY), -incomingRank);

    std::vector<util::array<int, 2>> testArrayRef(gridSize, {thisRank, -thisRank});
    std::fill(begin(testArrayRef) + gridSize - nex, end(testArrayRef),
              util::array<int, 2>{incomingRank, -incomingRank});

    std::vector<uint8_t> refU8(gridSize, thisRank);
    std::fill(begin(refU8) + gridSize - nex, end(refU8), incomingRank);

    EXPECT_EQ(refX, x);
    EXPECT_EQ(refY, y);
    EXPECT_EQ(testArrayRef, testArray);
    EXPECT_EQ(refU8, uint8Array);
}

TEST(GlobalDomain, exchangeCyclicNeighbors)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    exchangeCyclicNeighbors(rank, numRanks);
    MPI_Barrier(MPI_COMM_WORLD);
}
