/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Test functions used to find peer ranks
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/traversal/peers.hpp"
#include "cstone/tree/cs_util.hpp"

#include "coord_samples/random.hpp"

using namespace cstone;

//! @brief reference peer search, all-all leaf comparison
template<class KeyType, class T>
static std::vector<int> findPeersAll2All(int myRank,
                                         const SfcAssignment<KeyType>& assignment,
                                         std::span<const KeyType> tree,
                                         const Box<T>& box,
                                         float invThetaEff)
{
    TreeNodeIndex firstIdx = findNodeAbove(tree.data(), nNodes(tree), assignment[myRank]);
    TreeNodeIndex lastIdx  = findNodeAbove(tree.data(), nNodes(tree), assignment[myRank + 1]);

    std::vector<Vec3<T>> boxCenter(nNodes(tree));
    std::vector<Vec3<T>> boxSize(nNodes(tree));
    for (TreeNodeIndex i = 0; i < TreeNodeIndex(nNodes(tree)); ++i)
    {
        IBox ibox                          = sfcIBox(sfcKey(tree[i]), sfcKey(tree[i + 1]));
        std::tie(boxCenter[i], boxSize[i]) = centerAndSize<KeyType>(ibox, box);
    }

    std::vector<int> peers(assignment.numRanks());
    for (TreeNodeIndex i = firstIdx; i < lastIdx; ++i)
        for (TreeNodeIndex j = 0; j < TreeNodeIndex(nNodes(tree)); ++j)
            if (!minVecMacMutual(boxCenter[i], boxSize[i], boxCenter[j], boxSize[j], box, invThetaEff))
            {
                peers[assignment.findRank(tree[j])] = 1;
            }

    std::vector<int> ret;
    for (int i = 0; i < int(peers.size()); ++i)
        if (peers[i] && i != myRank) { ret.push_back(i); }

    return ret;
}

template<class KeyType>
static void findMacPeers64grid(int rank, float theta, BoundaryType pbc, int /*refNumPeers*/)
{
    Box<double> box{-1, 1, pbc};
    Octree<KeyType> octree;
    auto leaves = makeUniformNLevelTree<KeyType>(64, 1);
    octree.update(leaves.data(), nNodes(leaves));

    SfcAssignment<KeyType> assignment(octree.numLeafNodes());
    for (int i = 0; i < octree.numLeafNodes() + 1; ++i)
    {
        assignment.set(i, leaves[i], 1);
    }

    std::vector<int> peers     = findPeersMac(rank, assignment, octree, box, invThetaVecMac(theta));
    std::vector<int> reference = findPeersAll2All(rank, assignment, octree.treeLeaves(), box, invThetaVecMac(theta));

    // EXPECT_EQ(refNumPeers, peers.size());
    EXPECT_EQ(peers, reference);
}

TEST(Peers, findMacGrid64)
{
    // just the surface
    findMacPeers64grid<unsigned>(0, 1.1, BoundaryType::open, 7);
    findMacPeers64grid<uint64_t>(0, 1.1, BoundaryType::open, 7);
}

TEST(Peers, findMacGrid64Narrow)
{
    findMacPeers64grid<unsigned>(0, 1.0, BoundaryType::open, 19);
    findMacPeers64grid<uint64_t>(0, 1.0, BoundaryType::open, 19);
}

TEST(Peers, findMacGrid64PBC)
{
    // just the surface + PBC, 26 six peers at the surface
    findMacPeers64grid<unsigned>(0, 1.1, BoundaryType::periodic, 26);
    findMacPeers64grid<uint64_t>(0, 1.1, BoundaryType::periodic, 26);
}

template<class KeyType>
static void findPeers()
{
    Box<double> box{-1, 1};
    int nParticles    = 100000;
    int bucketSize    = 64;
    int numRanks      = 50;
    float invThetaEff = invThetaVecMac(0.5f);

    auto particleKeys   = makeRandomGaussianKeys<KeyType>(nParticles);
    auto [tree, counts] = computeOctree<KeyType>(particleKeys, bucketSize);

    Octree<KeyType> octree;
    octree.update(tree.data(), nNodes(tree));

    auto assignment = makeSfcAssignment(numRanks, counts, tree.data());

    int probeRank             = numRanks / 2;
    std::vector<int> peersDtt = findPeersMac(probeRank, assignment, octree, box, invThetaEff);
    std::vector<int> peersStt = findPeersMacStt(probeRank, assignment, octree, box, invThetaEff);
    std::vector<int> peersA2A = findPeersAll2All<KeyType>(probeRank, assignment, tree, box, invThetaEff);
    EXPECT_EQ(peersDtt, peersStt);
    EXPECT_EQ(peersDtt, peersA2A);

    // check for mutuality
    for (int peerRank : peersDtt)
    {
        std::vector<int> peersOfPeerDtt = findPeersMac(peerRank, assignment, octree, box, invThetaEff);

        // std::vector<int> peersOfPeerStt = findPeersMacStt(peerRank, assignment, octree, box, invThetaEff);
        // EXPECT_EQ(peersDtt, peersStt);
        std::vector<int> peersOfPeerA2A = findPeersAll2All<KeyType>(peerRank, assignment, tree, box, invThetaEff);
        EXPECT_EQ(peersOfPeerDtt, peersOfPeerA2A);

        // the peers of the peers of the probeRank have to have probeRank as peer
        EXPECT_TRUE(std::find(begin(peersOfPeerDtt), end(peersOfPeerDtt), probeRank) != end(peersOfPeerDtt));
    }
}

TEST(Peers, find)
{
    findPeers<unsigned>();
    findPeers<uint64_t>();
}
