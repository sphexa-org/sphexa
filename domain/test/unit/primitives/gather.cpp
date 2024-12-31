/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Test cpu gather functionality
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>

#include "gtest/gtest.h"

#include "cstone/primitives/gather.hpp"

using namespace cstone;

TEST(GatherCpu, sortInvert)
{
    std::vector<int> keys{2, 1, 5, 4};

    // the ordering that sorts keys is {1,0,3,2}
    std::vector<int> values(keys.size());
    std::iota(begin(values), end(values), 0);

    sort_by_key(begin(keys), end(keys), begin(values));

    std::vector<int> reference{1, 0, 3, 2};
    EXPECT_EQ(values, reference);
}

template<class ValueType, class KeyType, class IndexType>
void CpuGatherTest()
{
    std::vector<KeyType> keys{0, 50, 10, 60, 20, 70, 30, 80, 40, 90};

    std::vector<unsigned> scratch, s0, s1;
    SfcSorter<std::vector<unsigned>> sorter(scratch);
    sorter.sequence(0, keys.size());
    sorter.sortByKey(std::span(keys), 0, s0, s1);

    {
        std::vector<KeyType> refCodes{0, 10, 20, 30, 40, 50, 60, 70, 80, 90};
        EXPECT_EQ(keys, refCodes);
    }

    std::vector<ValueType> values{-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    std::vector<ValueType> probe = values;
    gatherCpu({sorter.getMap(), keys.size()}, values.data() + 2, probe.data() + 2);
    std::vector<ValueType> reference{-2, -1, 0, 2, 4, 6, 8, 1, 3, 5, 7, 9, 10, 11};

    EXPECT_EQ(probe, reference);
}

TEST(GatherCpu, CpuGather)
{
    CpuGatherTest<float, unsigned, unsigned>();
    CpuGatherTest<float, uint64_t, unsigned>();
    CpuGatherTest<double, unsigned, unsigned>();
    CpuGatherTest<double, uint64_t, unsigned>();
}
