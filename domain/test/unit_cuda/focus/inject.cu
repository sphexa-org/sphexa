/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Cornerstone octree GPU testing
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#include <vector>

#include "gtest/gtest.h"

#include "cstone/focus/inject.hpp"

using namespace cstone;

TEST(FocusGpu, injectKeysGpu)
{
    using KeyType = uint64_t;

    DeviceVector<KeyType> leaves        = std::vector<KeyType>{0, 64};
    DeviceVector<KeyType> mandatoryKeys = std::vector<KeyType>{0, 32, 64};

    DeviceVector<KeyType> keyScratch;
    DeviceVector<TreeNodeIndex> s1, s2;

    injectKeysGpu(leaves, {mandatoryKeys.data(), mandatoryKeys.size()}, keyScratch, s1, s2);

    DeviceVector<KeyType> ref = std::vector<KeyType>{0, 8, 16, 24, 32, 40, 48, 56, 64};

    EXPECT_EQ(leaves, ref);
}
