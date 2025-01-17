/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/primitives/concat_vector.hpp"

using namespace cstone;

template<class T>
auto testConstView(const ConcatVector<T>& v)
{
    return v.view();
}

TEST(PackBuffers, concatVector)
{
    ConcatVector<int> v;
    v.reindex({1, 2, 3});

    auto modView = v.view();
    std::iota(modView[0].begin(), modView[0].end(), 10);
    std::iota(modView[1].begin(), modView[1].end(), 20);
    std::iota(modView[2].begin(), modView[2].end(), 30);

    auto constView = testConstView(v);
    EXPECT_EQ(constView[2][0], 30);

    ConcatVector<int> v_cpy;
    copy(v, v_cpy);
    auto cpyView = v_cpy.view();
    EXPECT_TRUE(std::equal(modView[0].begin(), modView[0].end(), cpyView[0].begin()));
    EXPECT_TRUE(std::equal(modView[1].begin(), modView[1].end(), cpyView[1].begin()));
    EXPECT_TRUE(std::equal(modView[2].begin(), modView[2].end(), cpyView[2].begin()));
}
