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
using namespace cstone::ijloop;

template<TupleOfPointers T>
void foo(T a)
{
    std::cout << std::get<0>(a);
}

template<class ThP>
struct ijFunc
{
    template<class Tc, class ParticleData>
    void operator()(ParticleData i, ParticleData j, Vec3<Tc> rij, std::remove_pointer_t<ThP> r2)
    {
    }
};

template<class Tc, class ThP, class Input, ParticlePairInteraction<Tc, ThP, Input> F>
void ijLoop(F f)
{

}

TEST(InterfaceConcept, derefTuple)
{
    static_assert(std::is_same_v<std::tuple<double>, DereferencedTuple<std::tuple<double*>>::type>);
}

TEST(InterfaceConcept, tuples)
{
    using Tc = double;
    using ThP = float*;
    using Input = std::tuple<double*>;
    using Output = std::tuple<double*>;

    static_assert(ijloop::detail::IsTupleOfPointers<Input>{});

    ijFunc<ThP> f;

    IJLoopDataset<Tc, ThP, Input, Output, ijFunc<ThP>> dataset;

    //ijLoop<Tc, ThP, Input>(f);
}
