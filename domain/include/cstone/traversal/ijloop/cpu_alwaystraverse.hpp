/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Neighbor search on CPU
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <algorithm>
#include <tuple>
#include <memory>

#include "cstone/findneighbors.hpp"
#include "cstone/traversal/groups.hpp"
#include "cstone/traversal/ijloop/common.hpp"
#include "cstone/tree/octree.hpp"

namespace cstone::ijloop
{

namespace cpu_always_traverse_neighborhood_detail
{
template<class Tc, class KeyType, class ThP>
struct CpuAlwaysTraverseNeighborhood
{
    OctreeNsView<Tc, KeyType> tree;
    Box<Tc> box = {0, 0};
    LocalIndex firstBody, lastBody;
    const Tc *x, *y, *z;
    ThP h;
    unsigned ngmax;

    template<class... In, class... Out, class Interaction, class Postamble>
    void ijLoop(std::tuple<In*...> const& input,
                std::tuple<Out*...> const& output,
                Interaction&& interaction,
                Postamble&& postamble) const
    {
        const auto constInput = makeConst(input);
#pragma omp parallel
        {
            std::unique_ptr<LocalIndex[]> neighbors = std::make_unique_for_overwrite<LocalIndex[]>(ngmax);

#pragma omp for
            for (LocalIndex i = firstBody; i < lastBody; ++i)
                jLoop(constInput, output, std::forward<Interaction>(interaction), std::forward<Postamble>(postamble), i,
                      neighbors.get());
        }
    }

    Statistics stats() const { return {.numBodies = lastBody - firstBody, .numBytes = 0}; }

    struct Subgroup
    {
        CpuAlwaysTraverseNeighborhood const& parent;
        GroupView groups;

        template<class... In, class... Out, class Interaction, class Postamble>
        void ijLoop(std::tuple<In*...> const& input,
                    std::tuple<Out*...> const& output,
                    Interaction&& interaction,
                    Postamble&& postamble) const
        {
            const auto constInput = makeConst(input);
#pragma omp parallel
            {
                std::unique_ptr<LocalIndex[]> neighbors = std::make_unique_for_overwrite<LocalIndex[]>(parent.ngmax);

#pragma omp for
                for (LocalIndex g = 0; g < groups.numGroups; ++g)
                    for (LocalIndex i = groups.groupStart[g]; i < groups.groupEnd[g]; ++i)
                        parent.jLoop(constInput, output, std::forward<Interaction>(interaction),
                                     std::forward<Postamble>(postamble), i, neighbors.get());
            }
        }
    };

    Subgroup subgroup(GroupView const& groups) const { return {*this, groups}; }

protected:
    template<class Input, class Output, class Interaction, class Postamble>
    void jLoop(Input&& input,
               Output&& output,
               Interaction&& interaction,
               Postamble&& postamble,
               const LocalIndex i,
               LocalIndex* neighbors) const
    {
        const auto iData  = loadParticleData(x, y, z, h, std::forward<Input>(input), i);
        const bool usePbc = requiresPbcHandling(box, iData);

        const unsigned nbs = std::min(findNeighbors(i, x, y, z, h, tree, box, ngmax, neighbors), ngmax);
        auto result        = interaction(iData, iData, Vec3<Tc>{0, 0, 0}, Tc(0));
        for (unsigned nb = 0; nb < nbs; ++nb)
        {
            const LocalIndex j = neighbors[nb];
            const auto jData   = loadParticleData(x, y, z, h, std::forward<Input>(input), j);

            const auto [ijPosDiff, distSq] = posDiffAndDistSq(usePbc, box, iData, jData);

            updateResult(result, interaction(iData, jData, ijPosDiff, distSq));
        }

        storeParticleData(std::forward<Output>(output), i, postamble(iData, unwrapModifiers(result)));
    }
};

} // namespace cpu_always_traverse_neighborhood_detail

struct CpuAlwaysTraverseNeighborhoodBuilder
{
    unsigned ngmax;

    template<class Tc, class KeyType, class ThP>
    cpu_always_traverse_neighborhood_detail::CpuAlwaysTraverseNeighborhood<Tc, KeyType, ThP>
    build(const OctreeNsView<Tc, KeyType>& tree,
          const Box<Tc>& box,
          const LocalIndex /* totalBodies */,
          const GroupView& groups,
          const Tc* const x,
          const Tc* const y,
          const Tc* const z,
          const ThP h) const
    {
        return {tree, box, groups.firstBody, groups.lastBody, x, y, z, h, ngmax};
    }
};

} // namespace cstone::ijloop
