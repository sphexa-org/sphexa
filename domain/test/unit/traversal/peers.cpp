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

#include <fstream>

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

    int maxCoord = 1u << maxTreeLevel<KeyType>{};
    auto ellipse = Vec3<T>{box.ilx(), box.ily(), box.ilz()} * box.maxExtent() * invThetaEff;
    auto pbc_t   = BoundaryType::periodic;
    auto pbc     = Vec3<int>{box.boundaryX() == pbc_t, box.boundaryY() == pbc_t, box.boundaryZ() == pbc_t} * maxCoord;

    std::vector<IBox> boxes(nNodes(tree));
    for (TreeNodeIndex i = 0; i < TreeNodeIndex(nNodes(tree)); ++i)
    {
        boxes[i] = sfcIBox(sfcKey(tree[i]), sfcKey(tree[i + 1]));
    }

    std::vector<int> peers(assignment.numRanks());
    for (TreeNodeIndex i = firstIdx; i < lastIdx; ++i)
        for (TreeNodeIndex j = 0; j < TreeNodeIndex(nNodes(tree)); ++j)
            if (!minMacMutualInt(boxes[i], boxes[j], ellipse, pbc)) { peers[assignment.findRank(tree[j])] = 1; }

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

    std::vector<int> peers     = findPeersMac(rank, assignment, octree.cdata(), box, invThetaMinToVec(theta));
    std::vector<int> reference = findPeersAll2All(rank, assignment, octree.treeLeaves(), box, invThetaMinToVec(theta));

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
    float invThetaEff = invThetaMinToVec(0.5f);

    auto particleKeys   = makeRandomGaussianKeys<KeyType>(nParticles);
    auto [tree, counts] = computeOctree<KeyType>(particleKeys, bucketSize);

    Octree<KeyType> octree;
    octree.update(tree.data(), nNodes(tree));

    auto assignment = makeSfcAssignment(numRanks, counts, tree.data());

    int probeRank             = numRanks / 2;
    std::vector<int> peersDtt = findPeersMac(probeRank, assignment, octree.cdata(), box, invThetaEff);
    std::vector<int> peersStt = findPeersMacStt(probeRank, assignment, octree, box, invThetaEff);
    std::vector<int> peersA2A = findPeersAll2All<KeyType>(probeRank, assignment, tree, box, invThetaEff);
    EXPECT_EQ(peersDtt, peersStt);
    EXPECT_EQ(peersDtt, peersA2A);

    // check for mutuality
    for (int peerRank : peersDtt)
    {
        std::vector<int> peersOfPeerDtt = findPeersMac(peerRank, assignment, octree.cdata(), box, invThetaEff);

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

// A few harder tests to catch the FP-round-off asymmetric case

static bool isSymmetric(std::vector<std::vector<int>> matrix)
{
    int numRanks = matrix[0].size();
    for (int i = 0; i < numRanks; ++i)
        for (int j = i; j < numRanks; ++j)
        {
            if (matrix[i][j] != matrix[j][i]) { return false; }
        }
    return true;
}

//! @brief return number of rank pairs missing in probe relative to ref
static int compareMatrices(std::vector<std::vector<int>> ref, std::vector<std::vector<int>> probe)
{
    int np1 = 0, np2 = 0, missed = 0, extra = 0;
    int numRanks = ref[0].size();
    for (int i = 0; i < numRanks; ++i)
        for (int j = i; j < numRanks; ++j)
        {
            np1 += ref[i][j];
            np2 += probe[i][j];
            if (ref[i][j] and not probe[i][j]) missed++;
            if (probe[i][j] and not ref[i][j]) extra++;
        }
    std::cout << "numPairs " << np1 << "/" << np2 << " missed " << missed << " extra " << extra << std::endl;
    return missed;
}

template<class KeyType, class T>
auto peerMatrix(const std::vector<KeyType>& leaves,
                const std::vector<KeyType>& assignmentKeys,
                Box<T> box,
                float invThetaEff)
{
    Octree<KeyType> octree;
    octree.update(leaves.data(), nNodes(leaves));

    int numRanks = assignmentKeys.size() - 1;
    SfcAssignment<KeyType> assignment(numRanks);
    for (int r = 0; r <= numRanks; ++r)
    {
        assignment.set(r, assignmentKeys[r], 0);
    }

    std::vector matrix(numRanks, std::vector<int>(numRanks));

    for (int i = 0; i < numRanks; ++i)
    {
        auto peers = findPeersMac(i, assignment, octree.cdata(), box, invThetaEff);
        for (auto j : peers)
        {
            matrix[i][j] = 1;
        }
    }
    return matrix;
}

//! @brief compute peer matrix with Mac traversal, putting the expansion centers in a random corner
template<class KeyType, class T>
auto vecMacMatrix(const std::vector<KeyType>& leaves,
                  const std::vector<KeyType>& assignmentKeys,
                  Box<T> box,
                  float invTheta)
{
    Octree<KeyType> octree;
    octree.update(leaves.data(), nNodes(leaves));

    TreeNodeIndex numNodes = octree.numTreeNodes();

    int numRanks = assignmentKeys.size() - 1;
    SfcAssignment<KeyType> assignment(numRanks);
    for (int r = 0; r <= numRanks; ++r)
    {
        assignment.set(r, assignmentKeys[r], 0);
    }

    std::vector<Vec3<T>> centers(numNodes), sizes(numNodes);
    nodeFpCenters(octree.nodeKeys(), centers.data(), sizes.data(), box);

    std::vector<Vec4<T>> c4(numNodes);

    T margin        = 0.99;
    auto randCorner = [margin] { return drand48() > 0.5 ? margin : -margin; };
    for (size_t i = 0; i < c4.size(); ++i)
    {
        c4[i][0] = centers[i][0] + randCorner() * sizes[i][0];
        c4[i][1] = centers[i][1] + randCorner() * sizes[i][1];
        c4[i][2] = centers[i][2] + randCorner() * sizes[i][2];
        c4[i][3] = 1;
    }
    setMac<T, KeyType>(octree.nodeKeys(), c4, invTheta, box);

    std::vector matrix(numRanks, std::vector<int>(numRanks));
    for (int i = 0; i < numRanks; ++i)
    {
        TreeNodeIndex iStart = findNodeAbove(leaves.data(), nNodes(leaves), assignment[i]);
        TreeNodeIndex iEnd   = findNodeAbove(leaves.data(), nNodes(leaves), assignment[i + 1]);
        std::vector<uint8_t> macs_internal(numNodes, 0);
        markMacs(octree.nodeKeys().data(), octree.childOffsets().data(), octree.parents().data(), c4.data(), box,
                 leaves.data() + iStart, iEnd - iStart, false, macs_internal.data());

        std::vector<uint8_t> macs(octree.numLeafNodes(), 0);
        gather(octree.internalOrder(), macs_internal.data(), macs.data());

        for (int j = 0; j < numRanks; ++j)
        {
            TreeNodeIndex jStart = findNodeAbove(leaves.data(), nNodes(leaves), assignment[j]);
            TreeNodeIndex jEnd   = findNodeAbove(leaves.data(), nNodes(leaves), assignment[j + 1]);
            matrix[i][j] = std::any_of(macs.begin() + jStart, macs.begin() + jEnd, [](auto x) { return x > 0; });
        }
    }
    return matrix;
}

template<class KeyType>
std::vector<KeyType> makeAssignment(int numRanks)
{
    auto ret      = initialDomainSplits<KeyType>(numRanks, 5);
    KeyType delta = ret[1] / 64;

    auto randInt = [] { return int(6 * (drand48() - 0.5)); };

    for (size_t i = 1; i < ret.size() - 1; ++i)
    {
        ret[i] += randInt() * delta;
    }
    return ret;
}

TEST(Peers, pairs_nograv)
{
    using KeyType = uint64_t;

    std::vector<KeyType> ak = makeAssignment<KeyType>(256);

    std::vector<KeyType> leaves = computeSpanningTree<KeyType>(ak);

    int numRanks = 256;
    SfcAssignment<KeyType> assignment(numRanks);
    for (int r = 0; r <= numRanks; ++r)
    {
        assignment.set(r, ak[r], 0);
    }

    Box<double> box(-0.028, 0.028, BoundaryType::periodic);

    float theta      = 1.0;
    auto mat_int_min = peerMatrix(leaves, ak, box, 1.0 / theta);

    // fp-based peers would not be symmetric
    EXPECT_TRUE(isSymmetric(mat_int_min));
}

TEST(Peers, pairs_grav)
{
    using KeyType = uint64_t;

    std::vector<KeyType> ak     = makeAssignment<KeyType>(256);
    std::vector<KeyType> leaves = computeSpanningTree<KeyType>(ak);

    int numRanks = 256;
    SfcAssignment<KeyType> assignment(numRanks);
    for (int r = 0; r <= numRanks; ++r)
    {
        assignment.set(r, ak[r], 0);
    }

    Box<double> box(-0.028, 0.028, -0.04, 0.04, -0.1, 0.1, BoundaryType::periodic, BoundaryType::periodic /*Z open */);

    float theta          = 0.5;
    float invThetaIntMin = invThetaMinToVec(theta);
    auto mat_int_min     = peerMatrix(leaves, ak, box, invThetaIntMin);

    auto mat_vecmac = vecMacMatrix(leaves, ak, box, 1.0f / theta);

    EXPECT_TRUE(isSymmetric(mat_int_min));
    EXPECT_EQ(compareMatrices(mat_vecmac, mat_int_min), 0);
}
