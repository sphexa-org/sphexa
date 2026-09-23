#pragma once

#include <compare>
#include <stdexcept>
#include <variant>

#include "cstone/execution.hpp"
#include "cstone/traversal/groups.hpp"
#include "cstone/traversal/ijloop/cpu_alwaystraverse.hpp"
#include "cstone/traversal/ijloop/cpu_fullnblist.hpp"
#include "sph/types.hpp"

namespace sph
{

template<class NeighborhoodBuilder, cstone::execution::Policy Exec>
using NeighborhoodDataType = decltype(std::declval<NeighborhoodBuilder>().build(
    std::declval<Exec>(), std::declval<cstone::OctreeNsView<sph::SphTypes::CoordinateType, sph::SphTypes::KeyType>>(),
    std::declval<cstone::Box<sph::SphTypes::CoordinateType>>(), 0, std::declval<cstone::GroupView>(),
    std::declval<sph::SphTypes::CoordinateType*>(), std::declval<sph::SphTypes::CoordinateType*>(),
    std::declval<sph::SphTypes::CoordinateType*>(), std::declval<sph::SphTypes::HydroType*>()));

template<class NeighborhoodBuilder, cstone::execution::Policy Exec>
using NeighborhoodSubgroupType = decltype(std::declval<NeighborhoodDataType<NeighborhoodBuilder, Exec>>().subgroup(
    std::declval<cstone::GroupView>()));

struct NeighborhoodData
{
    NeighborhoodData() {}

    void disableNeighborLists() { useNeighborLists = false; }

    template<class Dataset, class T>
    void build(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box, bool /* subgroups */)
    {
        neighborhood.emplace<0>();

        std::variant<cstone::ijloop::CpuAlwaysTraverseNeighborhoodBuilder,
                     cstone::ijloop::CpuFullNbListNeighborhoodBuilder>
            builder;

        if (useNeighborLists)
            builder = cstone::ijloop::CpuFullNbListNeighborhoodBuilder{d.ngmax};
        else
            builder = cstone::ijloop::CpuAlwaysTraverseNeighborhoodBuilder{d.ngmax};

        std::visit(
            [&](auto const& nb)
            {
                neighborhood = nb.build(cstone::execution::cpu, d.treeView, box, d.size(), groups, d.x.data(),
                                        d.y.data(), d.z.data(), d.h.data());
            },
            builder);
    }

    template<class... Args>
    void ijLoop(cstone::ijloop::IjLoopData<Args...> ijData) const
    {
        std::visit([&](auto const& nb) { nb.ijLoop(std::move(ijData)); }, neighborhood);
    }

private:
    std::variant<NeighborhoodDataType<cstone::ijloop::CpuAlwaysTraverseNeighborhoodBuilder, cstone::execution::Cpu>,
                 NeighborhoodDataType<cstone::ijloop::CpuFullNbListNeighborhoodBuilder, cstone::execution::Cpu>>
         neighborhood;
    bool useNeighborLists = true;
};

} // namespace sph
