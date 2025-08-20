/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Test peer detection performance
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <chrono>
#include <iostream>

#include "coord_samples/random.hpp"
#include "cstone/traversal/peers.hpp"

using namespace cstone;

int main()
{
    using KeyType = uint64_t;
    Box<double> box{-1, 1};

    LocalIndex nParticles = 20000;
    int bucketSize        = 1;
    int numRanks          = 50;

    auto keys = makeRandomGaussianKeys<KeyType>(nParticles);

    auto [treeLeaves, counts] = computeOctree<KeyType>(keys, bucketSize);
    Octree<KeyType> octree;
    octree.update(treeLeaves.data(), nNodes(treeLeaves));

    auto assignment = makeSfcAssignment(numRanks, counts, treeLeaves.data());
    int probeRank   = numRanks / 2;

    auto tp0                  = std::chrono::high_resolution_clock::now();
    std::vector<int> peersDtt = findPeersMac(probeRank, assignment, octree.cdata(), box, invThetaMinMac(0.5f));
    auto tp1                  = std::chrono::high_resolution_clock::now();

    double t2 = std::chrono::duration<double>(tp1 - tp0).count();
    std::cout << "find peers: " << t2 << " numPeers: " << peersDtt.size() << std::endl;
}
