/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Utility tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <numeric>
#include <string>
#include <vector>
#include "gtest/gtest.h"

#include "cstone/util/aligned_alloc.hpp"
#include "cstone/util/noinit_alloc.hpp"
#include "cstone/util/tuple_util.hpp"

using namespace util;

template<class T>
using AllocatorType = DefaultInitAdaptor<T, AlignedAllocator<T, 64>>;

TEST(Utils, TupleMap)
{
    auto testee = std::tuple(42, std::string("hello"));

    EXPECT_EQ(tupleMap([](auto) { return 0; }, testee), std::tuple(0, 0));
    EXPECT_EQ(tupleMap([](auto x) { return x + x; }, testee), std::tuple(84, "hellohello"));

    auto testee2 = std::tuple(-42, std::string(" world"));
    EXPECT_EQ(tupleMap([](auto x, auto y) { return x + y; }, testee, testee2), std::tuple(0, "hello world"));

    EXPECT_EQ(tupleMap(
                  [](auto& x, auto& y)
                  {
                      auto oldX = x;
                      std::swap(x, y);
                      return oldX;
                  },
                  testee, testee2),
              std::tuple(42, std::string("hello")));
    EXPECT_EQ(testee, std::tuple(-42, std::string(" world")));
    EXPECT_EQ(testee2, std::tuple(42, std::string("hello")));
}

TEST(Utils, ForEachTuple)
{
    std::vector<int> a{90};
    std::vector<double> b{3.14};

    auto tup = std::tie(a, b);

    for_each_tuple([](auto& v) { v.push_back(2); }, tup);

    EXPECT_EQ(a.size(), 2);
    EXPECT_EQ(b.size(), 2);
}

TEST(Utils, TupleReverse)
{
    std::vector<int> a{90};
    std::vector<double> b{3.14};
    auto tup = std::tie(a, b);

    auto revTup = reverse(tup);

    EXPECT_EQ(std::get<1>(revTup)[0], a[0]);

    a.push_back(91);

    EXPECT_EQ(std::get<1>(revTup)[1], a[1]);
}

TEST(Utils, ForEachTupleRvalue)
{
    std::vector<int> a{90};
    std::vector<double> b{3.14};

    auto tup = std::tie(a, b);

    for_each_tuple([](auto& v) { v.push_back(2); }, reverse(tup));

    EXPECT_EQ(a.size(), 2);
    EXPECT_EQ(b.size(), 2);
}

/*! @brief noinit allocation test
 *
 * Heap memory pages do get initialized to zero by the OS the first time
 * they are given to the process.
 * Therefore, to see any effects, i.e. vectors with non-zero elements after construction,
 * we first have to allocate a couple of pages and fill them with non-zero data.
 * Then we stand a good change that the final vector will get a page previously touched
 * by the process, which will contain non-zero data from previous allocations.
 */
TEST(Utils, AlignedNoInitAlloc)
{
    for (int i = 0; i < 100; ++i)
    {
        std::vector<double, AllocatorType<double>> v(512);
        std::iota(v.begin(), v.end(), 0);
    }

    // most likely allocated with previously touched pages
    std::vector<double, AllocatorType<double>> v(10);

    // content of v is uninitialized (not all zeros)
    EXPECT_NE(std::count(v.begin(), v.end(), 0.0), v.size());

    // address is 64-byte aligned
    EXPECT_EQ(reinterpret_cast<size_t>(v.data()) % 64, 0);

    // init to zero still works if explicitly specified
    std::vector<double, AllocatorType<double>> z(10, 0);
    EXPECT_EQ(std::count(z.begin(), z.end(), 0.0), z.size());
    EXPECT_EQ(reinterpret_cast<size_t>(z.data()) % 64, 0);

    // std::copy(v.begin(), v.end(), std::ostream_iterator<double>(std::cout, " "));
    // std::cout << " address: " << v.data() << std::endl;
}

TEST(Utils, discardLastElement)
{
    {
        auto tup   = std::make_tuple(1, 2, 3);
        auto probe = discardLastElement(tup);
        auto ref   = std::make_tuple(1, 2);
        EXPECT_EQ(probe, ref);
    }
    {
        // test that lvalue reference elements are passed on as lvalue references
        std::vector<int> v1{1, 2, 3, 4}, v2{5, 6, 7};
        auto probe = discardLastElement(std::tie(v1, v2));
        EXPECT_EQ(probe, std::tie(v1));
        EXPECT_EQ(std::get<0>(probe).data(), v1.data());
    }
}
