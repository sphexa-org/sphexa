/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief GPU SFC sorter unit tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#include <vector>

#include "gtest/gtest.h"

#include <cstone/cuda/device_vector.h>
#include "cstone/primitives/gather.cuh"

using namespace cstone;

TEST(SfcSorterGpu, shiftMapLeft)
{
    using KeyType   = unsigned;
    using IndexType = unsigned;

    DeviceVector<KeyType> keys = std::vector<KeyType>{2, 1, 5, 4};

    DeviceVector<IndexType> obuf, keyBuf, valBuf;
    GpuSfcSorter<IndexType, DeviceVector<unsigned>> sorter(obuf);

    sorter.setMapFromCodes<KeyType>({keys.data(), keys.size()}, 0, keyBuf, valBuf);
    // map is [1 0 3 2]

    {
        DeviceVector ref = std::vector<IndexType>{1, 0, 3, 2};
        EXPECT_EQ(obuf, ref);
    }

    sorter.extendMap(-1, keyBuf);

    {
        DeviceVector ref = std::vector<IndexType>{0, 2, 1, 4, 3};
        EXPECT_EQ(obuf, ref);
    }
}

TEST(SfcSorterGpu, shiftMapRight)
{
    using KeyType   = unsigned;
    using IndexType = unsigned;

    DeviceVector<KeyType> keys = std::vector<KeyType>{2, 1, 5, 4};

    DeviceVector<IndexType> obuf, keyBuf, valBuf;
    GpuSfcSorter<IndexType, DeviceVector<unsigned>> sorter(obuf);

    sorter.setMapFromCodes<KeyType>({keys.data(), keys.size()}, 0, keyBuf, valBuf);
    // map is [1 0 3 2]

    sorter.extendMap(1, keyBuf);
    {
        DeviceVector ref = std::vector<IndexType>{1, 0, 3, 2, 4};
        EXPECT_EQ(obuf, ref);
    }
}
