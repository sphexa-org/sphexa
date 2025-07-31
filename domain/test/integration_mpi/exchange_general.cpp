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
template<class KeyType, class T, template<class> class sfcKeyType>
static void generalExchangeRandomGaussian(int thisRank, int numRanks, const Box<T>& box)
{
    const LocalIndex numParticles = 1000;
    unsigned bucketSize           = 64;
    unsigned bucketSizeLocal      = 16;
    float theta                   = 10.0;
    float invThetaEff             = invThetaMinMac(theta);

    const auto mixDBits = getBoxMixDimensionBits<T, KeyType, Box<T>>(box);
    const bool useMixD = (mixDBits.bx != maxTreeLevel<KeyType>{} ||
                          mixDBits.by != maxTreeLevel<KeyType>{} ||
                          mixDBits.bz != maxTreeLevel<KeyType>{});

    // ******************************
    // identical data on all ranks

    // common pool of coordinates, identical on all ranks
    RandomCoordinates<T, sfcKeyType<KeyType>> coords =
        useMixD
            ? RandomCoordinates<T, sfcKeyType<KeyType>>{numRanks * numParticles, box, 42, mixDBits.bx, mixDBits.by,
                                                              mixDBits.bz}
            : RandomCoordinates<T, sfcKeyType<KeyType>>{numRanks * numParticles, box};

    auto [tree, counts] = computeOctree<KeyType>(coords.particleKeys(), bucketSize);

    Octree<KeyType> domainTree;
    domainTree.update(tree.data(), nNodes(tree));

    auto assignment = makeSfcAssignment(numRanks, counts, tree.data());

    // *******************************

    auto peers = findPeersMac(thisRank, assignment, domainTree, box, invThetaEff);

    std::cout << "[GeneralFocusExchange] rank " << thisRank << " peers: ";
    for (auto r : peers)
    {
        std::cout << r << " ";
    }
    std::cout << std::endl;

    KeyType focusStart = assignment[thisRank];
    KeyType focusEnd   = assignment[thisRank + 1];

    // locate particles assigned to thisRank
    auto firstAssignedIndex = findNodeAbove(coords.particleKeys().data(), coords.particleKeys().size(), focusStart);
    auto lastAssignedIndex  = findNodeAbove(coords.particleKeys().data(), coords.particleKeys().size(), focusEnd);
    std::cout << "[GeneralFocusExchange] rank " << thisRank
              << " firstAssignedIndex: " << firstAssignedIndex << " lastAssignedIndex: " << lastAssignedIndex
              << std::endl;

    // extract a slice of the common pool, each rank takes a different slice, but all slices together
    // are equal to the common pool
    std::vector<T> x(coords.x().begin() + firstAssignedIndex, coords.x().begin() + lastAssignedIndex);
    std::vector<T> y(coords.y().begin() + firstAssignedIndex, coords.y().begin() + lastAssignedIndex);
    std::vector<T> z(coords.z().begin() + firstAssignedIndex, coords.z().begin() + lastAssignedIndex);

    // Now build the focused tree using distributed algorithms. Each rank only uses its slice.
    std::vector<KeyType> particleKeys(lastAssignedIndex - firstAssignedIndex);
    if (useMixD)
    {
        computeSfcMixDKeys(x.data(), y.data(), z.data(), SfcMixDKindPointer(particleKeys.data()), x.size(), box,
                           mixDBits.bx, mixDBits.by, mixDBits.bz);
    }
    else
    {
        computeSfcKeys(x.data(), y.data(), z.data(), sfcKindPointer(particleKeys.data()), x.size(), box);
    }

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
            if (testCounts[i] != 0) {
                std::cout << "[GeneralFocusExchange] rank " << thisRank << " node " << i << " count: " << testCounts[i]
                        << " nodeStart: " << std::oct << nodeStart << std::dec << " nodeEnd: " << std::oct << nodeEnd
                        << std::dec << std::endl;
            }
        }
    }

    // calculate sum of testCounts
    unsigned testCountsCount{};
    for (auto count : testCounts)
    {
        if (count != -1) { testCountsCount += count; }
    }
    std::cout << "[GeneralFocusExchange] rank " << thisRank << " testCountsCount: " << testCountsCount << std::endl;
    // calculate sum of testCountsCount
    unsigned globalTestCountsCount{};
    MPI_Allreduce(&testCountsCount, &globalTestCountsCount, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    std::cout << "[GeneralFocusExchange] globalTestCountsCount: " << globalTestCountsCount << std::endl;
    const auto totalNumParticles = numRanks * numParticles;
    EXPECT_EQ(globalTestCountsCount, totalNumParticles);

    const auto print_rank = 0;

    if (thisRank == print_rank) {
        std::cout << "[GeneralFocusExchange] rank " << thisRank << " testCounts before upsweep: ";
        for (auto count : testCounts)
        {
            if (count != -1) std::cout << count << " ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }

    upsweep({octree.levelRange, maxTreeLevel<KeyType>{} + 2}, {octree.childOffsets, size_t(octree.numNodes)},
            testCounts.data(), NodeCount<unsigned>{});

    // if (thisRank == print_rank) {
        std::cout << "[GeneralFocusExchange] rank " << thisRank << " testCounts after upsweep: ";
        for (auto count : testCounts)
        {
            if (count != -1) std::cout << count << " ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    // }

    std::vector<int> scratch;
    focusTree.template peerExchange<unsigned>(testCounts, static_cast<int>(P2pTags::focusPeerCounts) + 2, scratch);
    if (thisRank == print_rank) {
        // difference here for rank 1
        std::cout << "[GeneralFocusExchange] rank " << thisRank << " testCounts after peerExchange: ";
        for (auto count : testCounts)
        {
            if (count != -1) std::cout << count << " ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }

    upsweep({octree.levelRange, maxTreeLevel<KeyType>{} + 2}, {octree.childOffsets, size_t(octree.numNodes)},
            testCounts.data(), NodeCount<unsigned>{});

    auto upsweepFunction = [](auto levelRange, auto childOffsets, auto M)
    { upsweep(levelRange, childOffsets, M, NodeCount<unsigned>{}); };
    globalFocusExchange<unsigned>(domainTree, focusTree, testCounts, upsweepFunction);

    if (thisRank == print_rank) {
        // no difference here for rank 1
        std::cout << "[GeneralFocusExchange] rank " << thisRank << " testCounts after globalFocusExchange: ";
        for (auto count : testCounts)
        {
            if (count != -1) std::cout << count << " ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }

    upsweep({octree.levelRange, maxTreeLevel<KeyType>{} + 2}, {octree.childOffsets, size_t(octree.numNodes)},
            testCounts.data(), NodeCount<unsigned>{});

    if (thisRank == print_rank) {
        std::cout << "[GeneralFocusExchange] rank " << thisRank << " testCounts after final upsweep: ";
        for (auto count : testCounts)
        {
            if (count != -1) std::cout << count << " ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }

    for (size_t i = 0; i < testCounts.size(); ++i)
    {
        if (testCounts[i] == -1)
        {
            std::cout << "[GeneralFocusExchange] rank " << thisRank << " node " << i << " testCount is -1" << std::endl;
        }
    }

    {
        for (size_t i = 0; i < testCounts.size(); ++i)
        {
            KeyType nodeStart = decodePlaceholderBit(octree.prefixes[i]);
            KeyType nodeEnd   = nodeStart + nodeRange<KeyType>(decodePrefixLength(octree.prefixes[i]) / 3);

            unsigned referenceCount = calculateNodeCount(nodeStart, nodeEnd, coords.particleKeys().data(),
                                                         coords.particleKeys().data() + coords.particleKeys().size(),
                                                         std::numeric_limits<unsigned>::max());
            if (testCounts[i] != -1 && testCounts[i] != referenceCount)
            {
                std::cout << "[GeneralFocusExchange] rank " << thisRank << " node " << i << " testCount: " << testCounts[i]
                          << " referenceCount: " << referenceCount << " nodeStart: " << std::oct << nodeStart << std::dec << " nodeEnd: " << std::oct
                          << nodeEnd << std::dec << std::endl;
            }
            // EXPECT_EQ(testCounts[i], referenceCount);
        }
    }

    EXPECT_EQ(testCounts[0], numRanks * numParticles);
}

TEST(GeneralFocusExchange, randomGaussian)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    generalExchangeRandomGaussian<unsigned, double, SfcKind>(rank, nRanks, {-1, 1});
    generalExchangeRandomGaussian<uint64_t, double, SfcKind>(rank, nRanks, {-1, 1});
    generalExchangeRandomGaussian<unsigned, float, SfcKind>(rank, nRanks, {-1, 1});
    generalExchangeRandomGaussian<uint64_t, float, SfcKind>(rank, nRanks, {-1, 1});
    generalExchangeRandomGaussian<unsigned, double, SfcMixDKind>(rank, nRanks, {0, 1, 0, 0.015625, 0, 0.00390625});
    generalExchangeRandomGaussian<uint64_t, double, SfcMixDKind>(rank, nRanks, {0, 1, 0, 0.015625, 0, 0.00390625});
    generalExchangeRandomGaussian<unsigned, float, SfcMixDKind>(rank, nRanks, {0, 1, 0, 0.015625, 0, 0.00390625});
    generalExchangeRandomGaussian<uint64_t, float, SfcMixDKind>(rank, nRanks, {0, 1, 0, 0.015625, 0, 0.00390625});
}

template<class KeyType, class T, template<class> class sfcKeyType>
static void generalExchangeSourceCenter(int thisRank, int numRanks, const Box<T>& box)
{
    const LocalIndex numParticles = 1000;
    unsigned bucketSize           = 64;
    unsigned bucketSizeLocal      = 16;
    float theta                   = 10.0;
    float invThetaEff             = invThetaMinMac(theta);

    const auto mixDBits = getBoxMixDimensionBits<T, KeyType>(box);
    const bool useMixD = (mixDBits.bx != maxTreeLevel<KeyType>{} ||
                          mixDBits.by != maxTreeLevel<KeyType>{} ||
                          mixDBits.bz != maxTreeLevel<KeyType>{});

    /*******************************/
    /* identical data on all ranks */

    // common pool of coordinates, identical on all ranks
    RandomGaussianCoordinates<T, sfcKeyType<KeyType>> coords =
        useMixD
            ? RandomGaussianCoordinates<T, sfcKeyType<KeyType>>{numRanks * numParticles, box, 42, mixDBits.bx, mixDBits.by, mixDBits.bz}
            : RandomGaussianCoordinates<T, sfcKeyType<KeyType>>{numRanks * numParticles, box};

    std::vector<T> globalMasses(numRanks * numParticles, 1.0 / (numRanks * numParticles));

    auto [tree, counts] = computeOctree(std::span(coords.particleKeys()), bucketSize);

    Octree<KeyType> domainTree;
    domainTree.update(tree.data(), nNodes(tree));

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
    if (useMixD)
    {
        computeSfcMixDKeys(x.data(), y.data(), z.data(), SfcMixDKindPointer(particleKeys.data()), x.size(), box,
                           mixDBits.bx, mixDBits.by, mixDBits.bz);
    }
    else
    {
        computeSfcKeys(x.data(), y.data(), z.data(), sfcKindPointer(particleKeys.data()), x.size(), box);
    }

    FocusedOctree<KeyType, T> focusTree(thisRank, numRanks, bucketSizeLocal);
    focusTree.converge(box, particleKeys, peers, assignment, tree, counts, invThetaEff);

    auto octree = focusTree.octreeViewAcc();

    focusTree.updateCenters(x.data(), y.data(), z.data(), m.data(), domainTree, box);
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

    generalExchangeSourceCenter<uint64_t, double, SfcKind>(rank, nRanks, {-1, 1});
    generalExchangeSourceCenter<unsigned, float, SfcKind>(rank, nRanks, {-1, 1});
    generalExchangeSourceCenter<uint64_t, double, SfcMixDKind>(rank, nRanks, {0, 1, 0, 0.015625, 0, 0.00390625});
    generalExchangeSourceCenter<unsigned, float, SfcMixDKind>(rank, nRanks, {0, 1, 0, 0.015625, 0, 0.00390625});
}
