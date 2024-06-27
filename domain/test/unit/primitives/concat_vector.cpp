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
    v.reindex({1, 1, 2, 2});

    auto modView  = v.view();
    modView[0][0] = 42;

    auto constView = testConstView(v);
    EXPECT_EQ(constView[0][0], 42);
}
