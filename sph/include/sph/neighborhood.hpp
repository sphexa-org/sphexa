#pragma once

#include <compare>
#include <stdexcept>
#include <variant>

#include "cstone/traversal/groups.hpp"
#include "cstone/traversal/ijloop/cpu_fullnblist.hpp"
#include "sph/types.hpp"

namespace sph
{

template<class NeighborhoodBuilder>
using NeighborhoodDataType = decltype(std::declval<NeighborhoodBuilder>().build(
    std::declval<cstone::OctreeNsView<sph::SphTypes::CoordinateType, sph::SphTypes::KeyType>>(),
    std::declval<cstone::Box<sph::SphTypes::CoordinateType>>(), 0, std::declval<cstone::GroupView>(),
    std::declval<sph::SphTypes::CoordinateType*>(), std::declval<sph::SphTypes::CoordinateType*>(),
    std::declval<sph::SphTypes::CoordinateType*>(), std::declval<sph::SphTypes::HydroType*>()));

template<class NeighborhoodBuilder>
using NeighborhoodSubgroupType =
    decltype(std::declval<NeighborhoodDataType<NeighborhoodBuilder>>().subgroup(std::declval<cstone::GroupView>()));

struct NeighborhoodData
{
    NeighborhoodData() {}

    template<class Dataset, class T>
    void build(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box, bool /* subgroups */)
    {
        neighborhood = {};

        auto builder = cstone::ijloop::CpuFullNbListNeighborhoodBuilder{d.ngmax};

        neighborhood = builder.build(d.treeView, box, d.size(), groups, d.x.data(), d.y.data(), d.z.data(), d.h.data());
    }

    template<class... Args>
    void ijLoop(Args&&... args) const
    {
        neighborhood.ijLoop(std::forward<Args>(args)...);
    }

private:
    NeighborhoodDataType<cstone::ijloop::CpuFullNbListNeighborhoodBuilder> neighborhood;
};

} // namespace sph
