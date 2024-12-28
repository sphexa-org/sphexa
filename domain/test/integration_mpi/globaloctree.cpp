/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Global octree build test
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>
#include <mpi.h>

#include <gtest/gtest.h>

#include "cstone/tree/update_mpi.hpp"
#include "cstone/tree/cs_util.hpp"

using namespace cstone;

template<class KeyType>
void buildTree(int rank)
{
    constexpr unsigned level      = 2;
    std::vector<KeyType> allCodes = makeNLevelGrid<KeyType>(level);
    std::vector<KeyType> codes{begin(allCodes) + rank * allCodes.size() / 2,
                               begin(allCodes) + (rank + 1) * allCodes.size() / 2};

    int bucketSize = 8;

    std::vector<KeyType> tree = makeRootNodeTree<KeyType>();
    std::vector<unsigned> counts{unsigned(codes.size())};
    while (!updateOctreeGlobal<KeyType>(codes, bucketSize, tree, counts))
        ;

    std::vector<KeyType> refTree = OctreeMaker<KeyType>{}.divide().makeTree();

    std::vector<unsigned> refCounts(8, 8);

    EXPECT_EQ(counts, refCounts);
    EXPECT_EQ(tree, refTree);
}

TEST(GlobalTree, basicRegularTree32)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    constexpr int thisExampleRanks = 2;

    if (nRanks != thisExampleRanks) throw std::runtime_error("this test needs 2 ranks\n");

    buildTree<unsigned>(rank);
    buildTree<uint64_t>(rank);
}
