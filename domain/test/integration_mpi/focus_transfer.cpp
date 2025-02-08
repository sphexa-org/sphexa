/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Assignment exchange test
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <gtest/gtest.h>

#include "cstone/focus/exchange_focus.hpp"
#include "cstone/tree/cs_util.hpp"

using namespace cstone;

/*! @brief simple tree exchange test with 2 ranks
 *
 * Each ranks has a regular level-2 grid with 64 elements as tree.
 */
template<class KeyType>
static void focusTransfer(int myRank, [[maybe_unused]] int numRanks)
{
    std::vector<KeyType> treeLeaves = makeUniformNLevelTree<KeyType>(64, 1);
    std::vector<unsigned> counts(nNodes(treeLeaves), 32);
    unsigned bucketSize = 64;

    // one of the nodes in the area to be handed over from rank 0 to rank 1 should be split based on counts
    TreeNodeIndex fullNode = findNodeAbove(treeLeaves.data(), treeLeaves.size(), pad(KeyType(032), 6));
    counts[fullNode]       = bucketSize + 1;

    std::vector<KeyType> buffer;

    // assignment before: 0 --- 040 --- nodeRange(0)
    //            after:  0 --- 032 --- nodeRange(0)

    KeyType a = 0;
    KeyType b = 0;
    KeyType c = 0;
    KeyType d = 0;
    if (myRank == 0)
    {
        a = 0;
        b = pad<KeyType>(4, 3);
        c = 0;
        d = pad<KeyType>(032, 2 * 3);
    }
    else if (myRank == 1)
    {
        a = pad<KeyType>(4, 3);
        b = nodeRange<KeyType>(0);
        c = pad<KeyType>(032, 2 * 3);
        d = nodeRange<KeyType>(0);
    }

    focusTransfer<KeyType>(treeLeaves, counts, 64, myRank, a, b, c, d, buffer);

    // rank 1 receives rebalanced treelet for [032 - 040] from rank 0
    if (myRank == 1)
    {
        EXPECT_EQ(buffer.size(), 13);
        EXPECT_EQ(buffer.front(), pad(KeyType(032), 6));
        EXPECT_EQ(buffer[1], pad(KeyType(0321), 9));
        EXPECT_EQ(buffer.back(), pad(KeyType(037), 6));
    }
}

TEST(FocusTransfer, simpleTest)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    focusTransfer<unsigned>(rank, numRanks);
    focusTransfer<uint64_t>(rank, numRanks);
}

template<class KeyType>
static void focusTransferNRanks(int myRank, int numRanks)
{
    using SignedKey                 = std::make_signed_t<KeyType>;
    std::vector<KeyType> treeLeaves = makeUniformNLevelTree<KeyType>(64, 1);
    std::vector<unsigned> counts(nNodes(treeLeaves), 32);
    unsigned bucketSize = 64;

    std::vector<KeyType> buffer;

    SignedKey sign  = (myRank % 2 == 1) ? 1 : -1;
    SignedKey delta = pad<KeyType>(02, 6);

    KeyType a = myRank * (nodeRange<KeyType>(0) / numRanks);
    KeyType b = (myRank + 1) * (nodeRange<KeyType>(0) / numRanks);
    KeyType c = (myRank == 0) ? 0 : a - sign * delta;
    KeyType d = (myRank == numRanks - 1) ? nodeRange<KeyType>(0) : b + sign * delta;

    if (myRank < numRanks) { focusTransfer<KeyType>(treeLeaves, counts, bucketSize, myRank, a, b, c, d, buffer); }

    if (myRank == 0) { EXPECT_EQ(buffer.size(), 0); }
    else if (myRank == numRanks - 1) { EXPECT_EQ(buffer.size(), 2); }
    else if (sign > 0) { EXPECT_EQ(buffer.size(), 4); }
    else { EXPECT_EQ(buffer.size(), 0); }
}

TEST(FocusTransfer, simpleTestNRanks)
{
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // round down to closest power of two
    int numRanksUse = 1u << (31u - countLeadingZeros(unsigned(numRanks)));

    focusTransferNRanks<unsigned>(rank, numRanksUse);
    focusTransferNRanks<uint64_t>(rank, numRanksUse);
}
