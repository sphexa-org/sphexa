/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Space filling curve octree assignment to ranks tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/domain/buffer_description.hpp"
#include "cstone/domain/domaindecomp.hpp"

using namespace cstone;

TEST(DomainDecomposition, uniformBins)
{
    {
        unsigned umax = std::numeric_limits<unsigned>::max();
        int numSplits = 2;
        std::vector<unsigned> counts{umax - 10, 5, 5, umax - 11, 5, 6};

        std::vector<TreeNodeIndex> bins(numSplits + 1);
        std::vector<unsigned> binCounts(numSplits);
        uniformBins(counts, bins, binCounts);

        std::vector<TreeNodeIndex> ref{0, 3, 6};
        std::vector<unsigned> refCnt{umax, umax};
        EXPECT_EQ(bins, ref);
        EXPECT_EQ(binCounts, refCnt);
    }
    {
        int numSplits = 2;
        std::vector<unsigned> counts{5, 5, 5, 15, 1, 0};

        std::vector<TreeNodeIndex> bins(numSplits + 1);
        std::vector<unsigned> binCounts(numSplits);
        uniformBins(counts, bins, binCounts);

        std::vector<TreeNodeIndex> ref{0, 3, 6};
        EXPECT_EQ(bins, ref);
    }
    {
        int numSplits = 2;
        std::vector<unsigned> counts{15, 0, 1, 5, 5, 5};

        std::vector<TreeNodeIndex> bins(numSplits + 1);
        std::vector<unsigned> binCounts(numSplits);
        uniformBins(counts, bins, binCounts);

        EXPECT_EQ(*std::min_element(binCounts.begin(), binCounts.end()), 15);
        EXPECT_EQ(*std::max_element(binCounts.begin(), binCounts.end()), 16);
    }
    {
        int numSplits = 7;
        std::vector<unsigned> counts{4, 3, 4, 3, 4, 3, 4, 3, 4, 3};

        std::vector<TreeNodeIndex> bins(numSplits + 1);
        std::vector<unsigned> binCounts(numSplits);
        uniformBins(counts, bins, binCounts);

        EXPECT_EQ(*std::min_element(binCounts.begin(), binCounts.end()), 3);
        EXPECT_EQ(*std::max_element(binCounts.begin(), binCounts.end()), 7);
    }
}

TEST(DomainDecomposition, makeSfcAssignment)
{
    using KeyType = uint64_t;
    std::vector<KeyType> csarray{0, 10, 20, 30, 40};
    std::vector<unsigned> counts{5, 5, 5, 5};

    auto a = makeSfcAssignment(2, counts, csarray.data());
    EXPECT_EQ(a[0], 0);
    EXPECT_EQ(a[1], 20);
    EXPECT_EQ(a[2], 40);

    std::vector<TreeNodeIndex> refOffsets{0, 2, 4};
    EXPECT_TRUE(std::equal(a.treeOffsets().begin(), a.treeOffsets().end(), refOffsets.begin()));

    std::vector<TreeNodeIndex> refNumNodesPerRank{2, 2};
    EXPECT_TRUE(std::equal(a.numNodesPerRank().begin(), a.numNodesPerRank().end(), refNumNodesPerRank.begin()));
}

//! @brief test that the SfcLookupKey can lookup the rank for a given code
TEST(DomainDecomposition, assignmentFindRank)
{
    using KeyType = uint64_t;
    int nRanks    = 3;
    SfcAssignment<KeyType> assignment(nRanks);
    assignment.set(0, 0, 0);
    assignment.set(1, 1, 0);
    assignment.set(2, 3, 0);
    assignment.set(3, 4, 0);

    EXPECT_EQ(0, assignment.findRank(0));
    EXPECT_EQ(1, assignment.findRank(1));
    EXPECT_EQ(1, assignment.findRank(2));
    EXPECT_EQ(2, assignment.findRank(3));
}

TEST(DomainDecomposition, limitBoundaryShifts)
{
    using KeyType = uint64_t;

    std::vector<KeyType> leaves{0, 5, 10, 15, 20, 25, 30};
    std::vector<unsigned> counts{1, 2, 3, 4, 5, 6};

    int numRanks = 3;
    SfcAssignment<KeyType> defAssignment, probe(numRanks);
    probe.set(0, 0, 3);
    probe.set(1, 10, 7);
    probe.set(2, 20, 11);
    probe.set(3, 30, 0);

    {
        auto probeCpy = probe;
        limitBoundaryShifts<KeyType>(defAssignment, probe, leaves, counts);
        EXPECT_TRUE(std::equal(probe.data(), probe.data() + probe.numRanks() + 1, probeCpy.data()));
    }
    {
        SfcAssignment<KeyType> newAssignment(numRanks);
        newAssignment.set(0, 0, 15);
        newAssignment.set(1, 25, 6);
        newAssignment.set(2, 30, 0);
        newAssignment.set(3, 30, 0);
        limitBoundaryShifts<KeyType>(probe, newAssignment, leaves, counts);
        EXPECT_EQ(newAssignment[0], 0);
        EXPECT_EQ(newAssignment[1], 20);
        EXPECT_EQ(newAssignment[2], 30);
        EXPECT_EQ(newAssignment[3], 30);
        EXPECT_EQ(newAssignment.totalCount(0), 10);
        EXPECT_EQ(newAssignment.totalCount(1), 11);
        EXPECT_EQ(newAssignment.totalCount(2), 0);

        std::vector<TreeNodeIndex> refNumNodesPerRank{4, 2, 0};
        std::vector<TreeNodeIndex> refTreeOffsets{0, 4, 6, 6};

        std::span numNodesPerRank = newAssignment.numNodesPerRank();
        EXPECT_TRUE(std::equal(numNodesPerRank.begin(), numNodesPerRank.end(), refNumNodesPerRank.begin()));

        std::span treeOffsets = newAssignment.treeOffsets();
        EXPECT_TRUE(std::equal(treeOffsets.begin(), treeOffsets.end(), refTreeOffsets.begin()));
    }
}

TEST(DomainDecomposition, createSendRanges)
{
    using KeyType = uint64_t;

    std::vector<KeyType> keys{0, 0, 1, 3, 4, 5, 6, 6, 9, 10};

    int numRanks = 2;
    SfcAssignment<KeyType> assignment(numRanks);
    LocalIndex ignored = -1;
    assignment.set(0, 0, ignored);
    assignment.set(1, 6, ignored);
    assignment.set(2, 10, ignored); // note: this excludes the last key

    // note: input keys need to be sorted
    auto sendList = createSendRanges<KeyType>(assignment, keys);

    EXPECT_EQ(sendList.count(0), 6);
    EXPECT_EQ(sendList.count(1), 3);

    EXPECT_EQ(sendList[0], 0);
    EXPECT_EQ(sendList[1], 6);
    EXPECT_EQ(sendList[2], 9);
}

TEST(DomainDecomposition, initialDomainSplit)
{
    using KeyType = uint64_t;

    auto ret = initialDomainSplits<KeyType>(3, 5);
    EXPECT_EQ(ret.front(), 0);
    EXPECT_EQ(ret.back(), nodeRange<KeyType>(0));
}