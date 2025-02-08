/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Focus exchange test
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <gtest/gtest.h>

#include "cstone/focus/exchange_focus.hpp"
#include "cstone/tree/cs_util.hpp"

using namespace cstone;

/*! @brief irregular treelet exchange with 2 ranks
 *
 * In this test, each rank has a regular level-3 grid in its assigned half
 * of the cube with 512/2 = 256 elements. Outside the assigned area,
 * the tree structure is irregular.
 */
template<class KeyType>
void exchangeFocusIrregular(int myRank, int numRanks)
{
    std::vector<KeyType> treeLeavesRef[numRanks], treeLeavesInitial[numRanks];
    std::vector<int> peers;
    std::vector<IndexPair<TreeNodeIndex>> peerFocusIndices(numRanks);

    // create reference trees
    {
        OctreeMaker<KeyType> octreeMaker;
        octreeMaker.divide();
        // regular level-3 grid in the half cube with x = 0...0.5
        for (int i = 0; i < 4; ++i)
        {
            octreeMaker.divide(i);
            for (int j = 0; j < 8; ++j)
            {
                octreeMaker.divide(i, j);
            }
        }
        // finer resolution at one location outside the regular grid + cells that don't exist on rank 1
        octreeMaker.divide(7).divide(7, 0);
        treeLeavesRef[0] = octreeMaker.makeTree();
        octreeMaker.divide(7, 0, 3);
        treeLeavesInitial[0] = octreeMaker.makeTree();
        EXPECT_EQ(treeLeavesRef[0].size() + 7, treeLeavesInitial[0].size());
    }
    {
        OctreeMaker<KeyType> octreeMaker;
        octreeMaker.divide();
        // regular level-3 grid in the half cube with x = 0.5...1
        for (int i = 4; i < 8; ++i)
        {
            octreeMaker.divide(i);
            for (int j = 0; j < 8; ++j)
            {
                octreeMaker.divide(i, j);
            }
        }
        // finer resolution at one location outside the regular grid
        octreeMaker.divide(1).divide(1, 6);
        treeLeavesRef[1]     = octreeMaker.makeTree();
        treeLeavesInitial[1] = treeLeavesRef[1];
    }

    std::vector<KeyType> treeLeaves = treeLeavesInitial[myRank];

    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(treeLeaves));
    updateInternalTree<KeyType>(treeLeaves, octree.data());

    if (myRank == 0)
    {
        peers.push_back(1);
        TreeNodeIndex peerStartIdx =
            std::lower_bound(begin(treeLeaves), end(treeLeaves), codeFromIndices<KeyType>({4})) - begin(treeLeaves);
        peerFocusIndices[1] = TreeIndexPair(peerStartIdx, nNodes(treeLeaves));
    }
    else
    {
        peers.push_back(0);
        TreeNodeIndex peerEndIdx =
            std::lower_bound(begin(treeLeaves), end(treeLeaves), codeFromIndices<KeyType>({4})) - begin(treeLeaves);
        peerFocusIndices[0] = TreeIndexPair(0, peerEndIdx);
    }

    std::vector<std::vector<KeyType>> treelets(numRanks);
    ConcatVector<TreeNodeIndex> treeletIdx;
    syncTreelets(peers, peerFocusIndices, octree, treeLeaves, treelets);
    indexTreelets<KeyType>(peers, octree.prefixes, octree.levelRange, treelets, treeletIdx);

    auto treeletView = treeletIdx.view();
    if (myRank == 0)
    {
        KeyType boundary = decodePlaceholderBit(KeyType(014));
        EXPECT_EQ(treeletView[1].size(), findNodeAbove(treeLeavesRef[1].data(), nNodes(treeLeavesRef[1]), boundary));
        // check that rank 0's interior tree matches the exterior tree of rank 1
        for (size_t i = 0; i < treeletView[1].size(); ++i)
        {
            KeyType tlKey = octree.prefixes[treeletView[1][i]];
            EXPECT_EQ(tlKey, encodePlaceholderBit2K(treeLeavesRef[1][i], treeLeavesRef[1][i + 1]));
        }
    }
    else
    {
        TreeNodeIndex peerStartIdx =
            std::lower_bound(begin(treeLeavesRef[0]), end(treeLeavesRef[0]), codeFromIndices<KeyType>({4})) -
            begin(treeLeavesRef[0]);

        TreeNodeIndex numNodesExtTreeRank0 = nNodes(treeLeavesRef[0]) - peerStartIdx;
        // size of rank 0's exterior tree should match interior treelet size on rank 1
        EXPECT_EQ(numNodesExtTreeRank0, treeletView[0].size());

        for (size_t i = 0; i < treeletView[0].size(); ++i)
        {
            const KeyType* refTree = &treeLeavesRef[0][peerStartIdx];
            KeyType tlKey          = octree.prefixes[treeletView[0][i]];
            EXPECT_EQ(tlKey, encodePlaceholderBit2K(refTree[i], refTree[i + 1]));
        }
    }
}

TEST(PeerExchange, irregularTree)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    constexpr int thisExampleRanks = 2;

    if (numRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    exchangeFocusIrregular<unsigned>(rank, numRanks);
    exchangeFocusIrregular<uint64_t>(rank, numRanks);
}

TEST(PeerExchange, arrayWrap)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    using Vec = util::array<uint64_t, 4>;

    if (rank == 0)
    {
        std::vector<Vec> buffer{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}};

        std::vector<MPI_Request> requests;
        mpiSendAsync(buffer.data(), buffer.size(), 1, 0, requests);
        MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUS_IGNORE);
    }
    if (rank == 1)
    {
        std::vector<Vec> buffer(3);
        mpiRecvSync(buffer.data(), buffer.size(), 0, 0, MPI_STATUS_IGNORE);

        std::vector<Vec> reference{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}};
        EXPECT_EQ(buffer, reference);
    }
}
