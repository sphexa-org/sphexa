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
#include <iostream>
#include <tuple>
#include <memory>

#include "cstone/execution.hpp"
#include "cstone/findneighbors.hpp"
#include "cstone/traversal/groups.hpp"
#include "cstone/traversal/ijloop/common.hpp"
#include "cstone/tree/octree.hpp"

namespace cstone::ijloop
{

namespace cpu_full_nb_list_neighborhood_detail
{

template<class Tc, class KeyType, class ThP>
struct CpuFullNbListNeighborhood
{
    OctreeNsView<Tc, KeyType> tree;
    Box<Tc> box = {0, 0};
    LocalIndex firstBody, lastBody;
    std::unique_ptr<LocalIndex[]> neighborsCount, neighbors;
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
#pragma omp parallel for simd
        for (LocalIndex i = firstBody; i < lastBody; ++i)
            jLoop(constInput, output, std::forward<Interaction>(interaction), std::forward<Postamble>(postamble), i);
    }

    Statistics stats() const
    {
        const LocalIndex numBodies = lastBody - firstBody;
        return {.numBodies = numBodies,
                .numBytes  = sizeof(LocalIndex) * numBodies + sizeof(LocalIndex) * numBodies * ngmax};
    }

    struct Subgroup
    {
        CpuFullNbListNeighborhood const& parent;
        GroupView groups;

        template<class... In, class... Out, class Interaction, class Postamble>
        void ijLoop(std::tuple<In*...> const& input,
                    std::tuple<Out*...> const& output,
                    Interaction&& interaction,
                    Postamble&& postamble) const
        {
            const auto constInput = makeConst(input);
#pragma omp parallel for
            for (LocalIndex g = 0; g < groups.numGroups; ++g)
#pragma omp simd
                for (LocalIndex i = groups.groupStart[g]; i < groups.groupEnd[g]; ++i)
                    parent.jLoop(constInput, output, std::forward<Interaction>(interaction),
                                 std::forward<Postamble>(postamble), i);
        }
    };

    Subgroup subgroup(GroupView const& groups) const { return {*this, groups}; }

protected:
    template<class Input, class Output, class Interaction, class Postamble>
    void
    jLoop(Input&& input, Output&& output, Interaction&& interaction, Postamble&& postamble, const LocalIndex i) const
    {
        const auto iData  = loadParticleData(x, y, z, h, std::forward<Input>(input), i);
        const bool usePbc = requiresPbcHandling(box, iData);

        const unsigned nbs = neighborsCount[i - firstBody];
        auto result        = interaction(iData, iData, Vec3<Tc>{0, 0, 0}, Tc(0));
        for (unsigned nb = 0; nb < nbs; ++nb)
        {
            const LocalIndex j = neighbors[(i - firstBody) * ngmax + nb];
            const auto jData   = loadParticleData(x, y, z, h, std::forward<Input>(input), j);

            const auto [ijPosDiff, distSq] = posDiffAndDistSq(usePbc, box, iData, jData);

            if (distSq < radiusSq(iData)) updateResult(result, interaction(iData, jData, ijPosDiff, distSq));
        }

        storeParticleData(std::forward<Output>(output), i, postamble(iData, unwrapModifiers(result)));
    }
};
} // namespace cpu_full_nb_list_neighborhood_detail

struct CpuFullNbListNeighborhoodBuilder
{
    unsigned ngmax;

    template<class Tc, class KeyType, class ThP>
    cpu_full_nb_list_neighborhood_detail::CpuFullNbListNeighborhood<Tc, KeyType, ThP>
    build(execution::Cpu,
          OctreeNsView<Tc, KeyType> tree,
          const Box<Tc>& box,
          const LocalIndex totalBodies,
          const GroupView& groups,
          const Tc* const x,
          const Tc* const y,
          const Tc* const z,
          const ThP h) const
    {
        using namespace cpu_full_nb_list_neighborhood_detail;

        const LocalIndex numBodies = groups.lastBody - groups.firstBody;

        CpuFullNbListNeighborhood<Tc, KeyType, ThP> nbList{
            tree,
            box,
            groups.firstBody,
            groups.lastBody,
            std::make_unique_for_overwrite<LocalIndex[]>(numBodies),
            std::make_unique_for_overwrite<LocalIndex[]>(std::size_t(numBodies) * ngmax),
            x,
            y,
            z,
            h,
            ngmax};

        using Th = std::remove_cvref_t<std::remove_pointer_t<ThP>>;
        ThP hExt = h;
        std::unique_ptr<Th[]> hExtData;
        if (tree.searchExtFactor != 1)
        {
            if constexpr (std::is_pointer_v<ThP>)
            {
                hExtData = std::make_unique_for_overwrite<Th[]>(totalBodies);
#pragma omp parallel for
                for (LocalIndex i = 0; i < totalBodies; ++i)
                    hExtData[i] = h[i] * tree.searchExtFactor;
                hExt = hExtData.get();
            }
            else { hExt = h * tree.searchExtFactor; }
            tree.searchExtFactor = 1;
        }

        unsigned maxNeighbors = 0;
#pragma omp parallel for reduction(max : maxNeighbors)
        for (LocalIndex i = 0; i < numBodies; ++i)
        {
            const unsigned neighborCount = std::min(
                findNeighbors(i + groups.firstBody, x, y, z, hExt, tree, box, ngmax, &nbList.neighbors[i * ngmax]),
                ngmax);
            nbList.neighborsCount[i] = neighborCount;
            maxNeighbors             = std::max(maxNeighbors, neighborCount);
        }

        if (maxNeighbors > ngmax)
        {
            std::cerr
                << "WARNING: overflow in neighbor list. Missing neighbors! Try to increase ngmax. Current ngmax is "
                << ngmax << ", but found up to " << maxNeighbors << " neighbor particles." << std::endl;
        }
        return nbList;
    }
};

} // namespace cstone::ijloop
