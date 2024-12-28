/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Test generic SFC functionality
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"
#include "cstone/sfc/sfc.hpp"

using namespace cstone;

TEST(SFC, commonNodePrefix)
{
    using T       = double;
    using KeyType = unsigned;

    Box<T> box(0, 1);

    {
        auto key = commonNodePrefix<HilbertKey<KeyType>>(Vec3<T>{0.7, 0.2, 0.2}, Vec3<T>{0.01, 0.01, 0.01}, box);
        EXPECT_EQ(key, 0000176134);
    }
    {
        auto key = commonNodePrefix<HilbertKey<KeyType>>(Vec3<T>{0.2393, 0.3272, 0.29372},
                                                         Vec3<T>{0.0012, 0.0011, 0.00098}, box);
        EXPECT_EQ(key, 0000104322);
    }
}

TEST(SFC, center)
{
    using T       = double;
    using KeyType = unsigned;

    Box<T> box(-1, 1);

    {
        // The exact center belongs to octant farthest from the origin
        T x           = 0.0;
        KeyType probe = sfc3D<HilbertKey<KeyType>>(x, x, x, box);
        KeyType ref   = sfc3D<HilbertKey<KeyType>>(1.0, 1.0, 1.0, box);
        EXPECT_EQ(octalDigit(probe, 1), octalDigit(ref, 1));
    }
    {
        // Center - epsilon should be in the octant closest to the origin
        T x           = -1e-40;
        KeyType probe = sfc3D<HilbertKey<KeyType>>(x, x, x, box);
        KeyType ref   = sfc3D<HilbertKey<KeyType>>(-1.0, -1.0, -1.0, box);
        EXPECT_EQ(octalDigit(probe, 1), octalDigit(ref, 1));
    }
}