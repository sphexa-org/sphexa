/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *               2024 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
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
