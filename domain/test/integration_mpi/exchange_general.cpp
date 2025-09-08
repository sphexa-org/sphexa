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

#include "cstone/focus/octree_focus_mpi.hpp"
#include "cstone/focus/source_center.hpp"
#include "cstone/traversal/peers.hpp"

#include "coord_samples/random.hpp"

using namespace cstone;

/*! @brief test for particle-count-exchange of distributed focused octrees
 *
 * First, all ranks create numRanks * numParticles random gaussian particle coordinates.
 * Since the random number generator is seeded with the same value on all ranks, all of them
 * will generate exactly the same numRanks * numParticles coordinates.
 *
 * This common coordinate set is then used to build a focus tree on each rank, using
 * non-communicating local algorithms to serve as the reference.
 * Each rank then takes the <thisRank>-th fraction of the common coordinate set and discards the other coordinates,
 * such that all ranks combined still have the original common set.
 * From the distributed coordinate set, the same focused trees are then built, but with distributed communicating
 * algorithms. This should yield the same tree on each rank as the local case,
 */
template<class KeyType, class T>
static void generalExchangeRandomGaussian(int thisRank, int numRanks)
{
    const LocalIndex numParticles = 1000;
    unsigned bucketSize           = 64;
    unsigned bucketSizeLocal      = 16;
    float theta                   = 10.0;
    float invThetaEff             = invThetaMinMac(theta);

    Box<T> box{-1, 1};

    // ******************************
    // identical data on all ranks

    // common pool of coordinates, identical on all ranks
    RandomGaussianCoordinates<T, SfcKind<KeyType>> coords(numRanks * numParticles, box);

    auto [tree, counts] = computeOctree<KeyType>(coords.particleKeys(), bucketSize);

    OctreeData<KeyType, CpuTag> domainTree_;
    domainTree_.resize(nNodes(tree));
    updateInternalTree<KeyType>(tree, domainTree_.data());
    auto domainTree   = domainTree_.cdata();
    domainTree.leaves = tree.data();

    auto assignment = makeSfcAssignment(numRanks, counts, tree.data());

    // *******************************

    auto peers = findPeersMac(thisRank, assignment, domainTree, box, invThetaEff);

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

    FocusedOctree<KeyType, T> focusTree(thisRank, numRanks, bucketSizeLocal);
    focusTree.converge(box, particleKeys, peers, assignment, tree, counts, invThetaEff);

    auto octree = focusTree.octreeViewAcc();
    std::vector<unsigned> testCounts(octree.numNodes, -1);

    for (TreeNodeIndex i = 0; i < octree.numNodes; ++i)
    {
        KeyType nodeStart = decodePlaceholderBit(octree.prefixes[i]);
        KeyType nodeEnd   = nodeStart + nodeRange<KeyType>(decodePrefixLength(octree.prefixes[i]) / 3);
        bool inFocus      = focusStart <= nodeStart && nodeEnd <= focusEnd;

        if (octree.childOffsets[i] == 0 && inFocus)
        {
            testCounts[i] =
                calculateNodeCount(nodeStart, nodeEnd, particleKeys.data(), particleKeys.data() + particleKeys.size(),
                                   std::numeric_limits<int>::max());
        }
    }

    upsweep(octree.levelRangeSpan(), octree.childOffsets, testCounts.data(), NodeCount<unsigned>{});

    std::vector<int> scratch;
    focusTree.template peerExchange<unsigned>(testCounts, static_cast<int>(P2pTags::focusPeerCounts) + 2, scratch);

    auto upsweepFunction = [](auto levelRange, auto childOffsets, auto M)
    { upsweep(levelRange, childOffsets, M, NodeCount<unsigned>{}); };
    focusTree.globalExchange(domainTree, std::span(testCounts), std::span<unsigned>{}, scratch, upsweepFunction);

    upsweep(octree.levelRangeSpan(), octree.childOffsets, testCounts.data(), NodeCount<unsigned>{});

    {
        for (size_t i = 0; i < testCounts.size(); ++i)
        {
            KeyType nodeStart = decodePlaceholderBit(octree.prefixes[i]);
            KeyType nodeEnd   = nodeStart + nodeRange<KeyType>(decodePrefixLength(octree.prefixes[i]) / 3);

            unsigned referenceCount = calculateNodeCount(nodeStart, nodeEnd, coords.particleKeys().data(),
                                                         coords.particleKeys().data() + coords.particleKeys().size(),
                                                         std::numeric_limits<unsigned>::max());
            EXPECT_EQ(testCounts[i], referenceCount);
        }
    }

    EXPECT_EQ(testCounts[0], numRanks * numParticles);
}

TEST(GeneralFocusExchange, randomGaussian)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    generalExchangeRandomGaussian<unsigned, double>(rank, nRanks);
    generalExchangeRandomGaussian<uint64_t, double>(rank, nRanks);
    generalExchangeRandomGaussian<unsigned, float>(rank, nRanks);
    generalExchangeRandomGaussian<uint64_t, float>(rank, nRanks);
}

template<class KeyType, class T>
static void generalExchangeSourceCenter(int thisRank, int numRanks)
{
    const LocalIndex numParticles = 1000;
    unsigned bucketSize           = 64;
    unsigned bucketSizeLocal      = 16;
    float theta                   = 10.0;
    float invThetaEff             = invThetaMinMac(theta);

    Box<T> box{-1, 1};

    /*******************************/
    /* identical data on all ranks */

    // common pool of coordinates, identical on all ranks
    RandomGaussianCoordinates<T, SfcKind<KeyType>> coords(numRanks * numParticles, box);
    std::vector<T> globalMasses(numRanks * numParticles, 1.0 / (numRanks * numParticles));

    auto [tree, counts] = computeOctree(std::span(coords.particleKeys()), bucketSize);

    OctreeData<KeyType, CpuTag> domainTree_;
    domainTree_.resize(nNodes(tree));
    updateInternalTree<KeyType>(tree, domainTree_.data());
    auto domainTree   = domainTree_.cdata();
    domainTree.leaves = tree.data();

    auto assignment = makeSfcAssignment(numRanks, counts, tree.data());

    /*******************************/

    auto peers = findPeersMac(thisRank, assignment, domainTree, box, invThetaEff);

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
    std::vector<T> m(globalMasses.begin() + firstAssignedIndex, globalMasses.begin() + lastAssignedIndex);

    // Now build the focused tree using distributed algorithms. Each rank only uses its slice.
    std::vector<KeyType> particleKeys(lastAssignedIndex - firstAssignedIndex);
    computeSfcKeys(x.data(), y.data(), z.data(), sfcKindPointer(particleKeys.data()), x.size(), box);

    FocusedOctree<KeyType, T> focusTree(thisRank, numRanks, bucketSizeLocal);
    focusTree.converge(box, particleKeys, peers, assignment, tree, counts, invThetaEff);

    auto octree = focusTree.octreeViewAcc();

    focusTree.updateCenters(x.data(), y.data(), z.data(), m.data(), domainTree);
    auto sourceCenter = focusTree.expansionCentersAcc();

    constexpr T tol = std::is_same_v<T, double> ? 1e-10 : 1e-4;
    {
        for (TreeNodeIndex i = 0; i < octree.numNodes; ++i)
        {
            KeyType nodeStart = decodePlaceholderBit(octree.prefixes[i]);
            KeyType nodeEnd   = nodeStart + nodeRange<KeyType>(decodePrefixLength(octree.prefixes[i]) / 3);

            LocalIndex startIndex =
                findNodeAbove(coords.particleKeys().data(), coords.particleKeys().size(), nodeStart);
            LocalIndex endIndex = findNodeAbove(coords.particleKeys().data(), coords.particleKeys().size(), nodeEnd);

            SourceCenterType<T> reference = massCenter<T>(coords.x().data(), coords.y().data(), coords.z().data(),
                                                          globalMasses.data(), startIndex, endIndex);

            EXPECT_NEAR(sourceCenter[i][0], reference[0], tol);
            EXPECT_NEAR(sourceCenter[i][1], reference[1], tol);
            EXPECT_NEAR(sourceCenter[i][2], reference[2], tol);
            EXPECT_NEAR(sourceCenter[i][3], reference[3], tol);
        }
    }
}

TEST(GeneralFocusExchange, sourceCenter)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    generalExchangeSourceCenter<uint64_t, double>(rank, nRanks);
    generalExchangeSourceCenter<unsigned, float>(rank, nRanks);
}
