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

#define USE_CUDA

#include "cstone/cuda/device_vector.h"
#include "cstone/focus/octree_focus_mpi.hpp"
#include "cstone/traversal/peers.hpp"

#include "coord_samples/random.hpp"

using namespace cstone;

TEST(GeneralFocusExchangeGpu, bareTreelet)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    if (rank > 1) { return; }

    std::vector<unsigned> counts(10);
    std::iota(counts.begin(), counts.begin() + 4, 100 * rank);

    std::vector<int> peers{rank ? 0 : 1};

    ConcatVector<TreeNodeIndex> gatherMaps;
    gatherMaps.reindex({2, 2});
    auto gmView              = gatherMaps.view();
    gmView[peers.front()][0] = 1;
    gmView[peers.front()][1] = 3;

    std::vector<TreeNodeIndex> scatterMap{8, 9};
    std::vector<IndexPair<TreeNodeIndex>> scatterSubRangePerRank{{0, 2}, {0, 2}};

    ConcatVector<TreeNodeIndex, DeviceVector> d_gatherMaps;
    copy(gatherMaps, d_gatherMaps);
    DeviceVector<TreeNodeIndex> d_scatterMap = scatterMap;
    DeviceVector<unsigned> d_counts          = counts;
    DeviceVector<char> scratch;

    auto d_gatherMapsView = static_cast<const ConcatVector<TreeNodeIndex, DeviceVector>&>(d_gatherMaps).view();
    exchangeTreeletGeneral<unsigned>(
        peers, d_gatherMapsView, {rawPtr(scatterSubRangePerRank), scatterSubRangePerRank.size()},
        {rawPtr(d_scatterMap), d_scatterMap.size()}, {rawPtr(d_counts), d_counts.size()}, 0, scratch);

    std::vector<unsigned> h_counts = toHost(d_counts);

    if (rank == 0)
    {
        EXPECT_EQ(h_counts[8], 101);
        EXPECT_EQ(h_counts[9], 103);
    }

    if (rank == 1)
    {
        EXPECT_EQ(h_counts[8], 1);
        EXPECT_EQ(h_counts[9], 3);
    }
}

//! @brief see test description of CPU version
template<class KeyType, class T>
static void generalExchangeRandomGaussian(int thisRank, int numRanks)
{
    const LocalIndex numParticles = 1000;
    unsigned bucketSize           = 64;
    unsigned bucketSizeLocal      = 16;
    float theta                   = 1.0;
    float invThetaEff             = invThetaMinMac(theta);

    Box<T> box{-1, 1};

    // ******************************
    // identical data on all ranks

    // common pool of coordinates, identical on all ranks
    RandomGaussianCoordinates<T, SfcKind<KeyType>> coords(numRanks * numParticles, box);

    auto [tree, counts] = computeOctree<KeyType>(coords.particleKeys(), bucketSize);

    Octree<KeyType> domainTree;
    domainTree.update(tree.data(), nNodes(tree));

    auto assignment = makeSfcAssignment(numRanks, counts, tree.data());

    // *******************************

    auto peers = findPeersMac(thisRank, assignment, domainTree.cdata(), box, invThetaEff);

    KeyType focusStart = assignment[thisRank];
    KeyType focusEnd   = assignment[thisRank + 1];

    // locate particles assigned to thisRank
    auto firstAssignedIndex = findNodeAbove(coords.particleKeys().data(), coords.particleKeys().size(), focusStart);
    auto lastAssignedIndex  = findNodeAbove(coords.particleKeys().data(), coords.particleKeys().size(), focusEnd);

    // extract a slice of the common pool, each rank takes a different slice, but all slices together
    // are equal to the common pool
    std::vector<T> x(coords.x().begin() + firstAssignedIndex, coords.x().begin() + lastAssignedIndex);
    std::vector<T> y(coords.y().begin() + firstAssignedIndex, coords.y().begin() + lastAssignedIndex);
    std::vector<T> z(coords.z().begin() + firstAssignedIndex, coords.z().begin() + lastAssignedIndex);

    // Now build the focused tree using distributed algorithms. Each rank only uses its slice.
    std::vector<KeyType> particleKeys(lastAssignedIndex - firstAssignedIndex);
    computeSfcKeys(x.data(), y.data(), z.data(), sfcKindPointer(particleKeys.data()), x.size(), box);

    DeviceVector<KeyType> d_keys = particleKeys;
    DeviceVector<T> d_scratch;
    std::span<const KeyType> d_keysView{rawPtr(d_keys), d_keys.size()};

    DeviceVector<KeyType> d_globTree = tree;
    std::span<const KeyType> d_globTreeView{rawPtr(d_globTree), d_globTree.size()};
    DeviceVector<unsigned> d_globCounts = counts;
    std::span<const unsigned> d_globCountsView{rawPtr(d_globCounts), d_globCounts.size()};

    FocusedOctree<KeyType, T, GpuTag> focusTree(thisRank, numRanks, bucketSizeLocal);
    focusTree.converge(box, d_keysView, peers, assignment, d_globTreeView, d_globCountsView, invThetaEff, d_scratch);

    auto d_countsView = focusTree.countsAcc();
    std::vector<unsigned> testCounts(d_countsView.size());
    memcpyD2H(d_countsView.data(), d_countsView.size(), testCounts.data());

    auto octreeView = focusTree.octreeViewAcc();
    std::vector<KeyType> prefixes(octreeView.numNodes);
    memcpyD2H(octreeView.prefixes, octreeView.numNodes, prefixes.data());

    {
        for (size_t i = 0; i < testCounts.size(); ++i)
        {
            KeyType nodeStart = decodePlaceholderBit(prefixes[i]);
            KeyType nodeEnd   = nodeStart + nodeRange<KeyType>(decodePrefixLength(prefixes[i]) / 3);

            unsigned referenceCount = calculateNodeCount(nodeStart, nodeEnd, coords.particleKeys().data(),
                                                         coords.particleKeys().data() + coords.particleKeys().size(),
                                                         std::numeric_limits<unsigned>::max());
            EXPECT_EQ(testCounts[i], referenceCount);
        }
    }

    EXPECT_EQ(testCounts[0], numRanks * numParticles);
}

TEST(GeneralFocusExchangeGpu, randomGaussian)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    generalExchangeRandomGaussian<uint64_t, double>(rank, nRanks);
}
