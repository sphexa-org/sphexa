/*
 * Cornerstone octree
 *
 * Copyright (c) 2026 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Neighborhood interface tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/traversal/ijloop/interface_concept.hpp"

using namespace cstone;

template<ijloop::TupleOfPointers T>
void foo(T a)
{
    std::cout << std::get<0>(a);
}

TEST(InterfaceConcept, tuples)
{
    using DptrTuple = std::tuple<double*>;
    using DTuple = std::tuple<double>;

    static_assert(ijloop::detail::IsTupleOfPointers<DptrTuple>{});
    static_assert(not ijloop::detail::IsTupleOfPointers<DTuple>{});

    foo(DptrTuple{});
}
