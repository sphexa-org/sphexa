/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Tests the SFC key exchange among peer ranks for halos
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>

#include <gtest/gtest.h>

#include "cstone/domain/exchange_keys.hpp"
#include "cstone/domain/layout.hpp"
#include "cstone/tree/cs_util.hpp"

using namespace cstone;

/*! @brief exchange keys and test correctness of resulting send array indices
 *
 *  n = myRank * 4;
 *
 *  On each rank:
 *
 *  node index      0     1     2     3     4     5     6     7
 *  tree:        | n+0 | n+2 | n+4 | n+5 | n+6 | n+7 | n+8 | n+10 | n+12 |
 *  counts:         2     2     1     1     1     1     2     2
 *  haloFlag:             X                             X
 *  assignment   |-----------|-----------------------|-------------------|
 *                 myRank-1           myRank             myRank+1
 *  layout:         0     0     2     3     4     5     6     8      8
 *
 *  Rank i wants the halos with keys [n+2:n+4] from rank i-1 and [n+8:n+10] from rank i+1, n = 4 * i,
 *  and sends out those requests.
 *
 *  On rank i, we receive a request from rank i-1.
 *  Rank i-1 wants particles with keys [(n-4)+8:(n-4)+10] = [n+4:n+6].
 *  On rank i, these keys are in node indices 2-4, and the layout positions of nodes 2-4 is also 2-4.
 *  ==> rank i will send particles with index in [2:4] to rank i-1
 *
 *  On rank i we receive a request from rank i+1.
 *  Rank i+1 wants [(n+4)+2:(n+4)+4] = [n+6:n+8], which on rank i is nodes 4-6, also at layout position 4-6.
 *  ==> rank i will send particles with index in [4:6] to rank i+1
 */
template<class KeyType>
void exchangeKeys(int myRank, int numRanks)
{
    std::vector<unsigned> counts{2, 2, 1, 1, 1, 1, 2, 2};
    std::vector<uint8_t> haloFlags{0, 1, 0, 0, 0, 0, 1, 0};
    std::vector<TreeNodeIndex> fakeLeaf2Internal{0, 1, 2, 3, 4, 5, 6, 7};

    std::vector<unsigned> layout(counts.size() + 1);

    KeyType o = myRank * 4;
    std::vector<KeyType> treeLeaves{o, o + 2, o + 4, o + 5, o + 6, o + 7, o + 8, o + 10, o + 12};

    std::vector<TreeIndexPair> assignment(numRanks);
    std::vector<int> peers;
    SendList reference(numRanks);

    if (myRank > 0)
    {
        assignment[myRank - 1] = TreeIndexPair(0, 2);
        peers.push_back(myRank - 1);
        reference[myRank - 1].addRange(2, 4);
    }

    assignment[myRank] = TreeIndexPair(2, 6);
    computeNodeLayout<false>(counts, haloFlags, fakeLeaf2Internal, assignment[myRank], layout);

    if (myRank < numRanks - 1)
    {
        assignment[myRank + 1] = TreeIndexPair(6, 8);
        peers.push_back(myRank + 1);
        reference[myRank + 1].addRange(4, 6);
    }

    SendList probe = exchangeRequestKeys<KeyType>(treeLeaves, assignment, peers, layout);

    EXPECT_EQ(probe, reference);
}

TEST(ExchangeKeys, fixedTreeSizePerRank)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    exchangeKeys<unsigned>(rank, numRanks);
}

/*! @brief
 * Rank 0's surface in rank 1's domain is just one node, while rank 1's surface in 0 is 6 nodes.
 *
 * Here we check that the halo flag exchange works correctly, specifically that the receive buffers
 * are large enough.
 */
template<class KeyType>
void unequalSurface(int myRank, int numRanks)
{
    std::vector<KeyType> treeLeaves = OctreeMaker<KeyType>().divide().makeTree();
    std::vector<unsigned> counts(nNodes(treeLeaves), 1);

    std::vector<uint8_t> haloFlags(nNodes(treeLeaves));
    std::vector<int> peers;

    std::vector<TreeIndexPair> assignment(numRanks);
    assignment[0] = TreeIndexPair(0, nNodes(treeLeaves) - 1);
    assignment[1] = TreeIndexPair(nNodes(treeLeaves) - 1, nNodes(treeLeaves));

    SendList reference(numRanks);

    std::vector<unsigned> layout(counts.size() + 1);

    if (myRank == 1)
    {
        haloFlags = std::vector<uint8_t>{1, 1, 0, 1, 1, 1, 1, 1};
        std::vector<TreeNodeIndex> fakeLeaf2Internal{0, 1, 2, 3, 4, 5, 6, 7};
        peers.push_back(0);
        computeNodeLayout<false>(counts, haloFlags, fakeLeaf2Internal, assignment[1], layout);
        EXPECT_EQ(layout[7], 6);
        reference[0].addRange(layout[7], layout[8]);

        std::vector<unsigned> refLayout{0, 1, 2, 2, 3, 4, 5, 6, 7};
        EXPECT_EQ(refLayout, layout);
    }

    if (myRank == 0)
    {
        haloFlags = std::vector<uint8_t>{0, 0, 0, 0, 0, 0, 0, 1};
        std::vector<TreeNodeIndex> fakeLeaf2Internal{0, 1, 2, 3, 4, 5, 6, 7};
        peers.push_back(1);
        computeNodeLayout<false>(counts, haloFlags, fakeLeaf2Internal, assignment[0], layout);
        reference[1].addRange(layout[0], layout[2]);
        reference[1].addRange(layout[3], layout[7]);
    }

    SendList probe = exchangeRequestKeys<KeyType>(treeLeaves, assignment, peers, layout);
    EXPECT_EQ(probe, reference);
}

TEST(ExchangeKeys, unequalSurface)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    unequalSurface<unsigned>(rank, numRanks);
}
