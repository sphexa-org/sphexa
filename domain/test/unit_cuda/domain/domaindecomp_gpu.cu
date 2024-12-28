/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Tests for GPU domain decomposition functions
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <thrust/device_vector.h>

#include "gtest/gtest.h"

#include "cstone/domain/domaindecomp_gpu.cuh"

using namespace cstone;
auto rp = [](auto& v) { return thrust::raw_pointer_cast(v.data()); };

/*! @brief test SendList creation from a SFC assignment
 *
 * This test creates an array with SFC keys and an
 * SFC assignment with SFC keys ranges.
 * CreateSendList then translates the code ranges into indices
 * valid for the SFC key array.
 */
template<class KeyType>
static void sendListMinimalGpu()
{
    std::vector<KeyType> tree{0, 2, 6, 8, 10};
    std::vector<KeyType> keys{0, 0, 1, 3, 4, 5, 6, 6, 7, 11};

    int numRanks = 2;
    SfcAssignment<KeyType> assignment(numRanks);
    assignment.set(0, tree[0], 0);
    assignment.set(1, tree[2], 0);
    assignment.set(2, tree[4], 0);

    thrust::device_vector<KeyType> d_keys = keys;
    thrust::device_vector<KeyType> d_searchKeys(numRanks + 1);
    thrust::device_vector<LocalIndex> d_indices(numRanks + 1);

    std::span<const KeyType> d_keyView{rp(d_keys), d_keys.size()};

    // note: key input needs to be sorted
    auto sendList = createSendRangesGpu(assignment, d_keyView, rp(d_searchKeys), rp(d_indices));

    EXPECT_EQ(sendList.count(0), 6);
    EXPECT_EQ(sendList.count(1), 3);

    EXPECT_EQ(sendList[0], 0);
    EXPECT_EQ(sendList[1], 6);
    EXPECT_EQ(sendList[2], 9);
}

TEST(DomainDecomposition, createSendListGpu)
{
    sendListMinimalGpu<unsigned>();
    sendListMinimalGpu<uint64_t>();
}