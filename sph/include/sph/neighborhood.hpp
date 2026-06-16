#pragma once

#include <compare>
#include <stdexcept>
#include <variant>

#include "cstone/traversal/groups.hpp"
#include "cstone/traversal/ijloop/cpu_alwaystraverse.hpp"
#include "cstone/traversal/ijloop/cpu_fullnblist.hpp"
#include "sph/types.hpp"

namespace sph
{

enum class NeighborhoodType
{
    alwaysTraverse,
    fullNeighborList,
    compressedFullNeighborList,
    compressedHalfNeighborList,
    clusteredNeighborList
};

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

    void setType(NeighborhoodType type)
    {
        if (type != NeighborhoodType::alwaysTraverse && type != NeighborhoodType::fullNeighborList)
            throw std::runtime_error("only always-traverse and full-neighbor-list are available on CPUs");
        neighborhoodType = type;
    }

    template<class Dataset, class T>
    void build(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box, bool /* subgroups */)
    {
        neighborhood.emplace<0>();

        std::variant<cstone::ijloop::CpuAlwaysTraverseNeighborhoodBuilder,
                     cstone::ijloop::CpuFullNbListNeighborhoodBuilder>
            builder;

        switch (neighborhoodType)
        {
            case NeighborhoodType::alwaysTraverse:
                builder = cstone::ijloop::CpuAlwaysTraverseNeighborhoodBuilder{d.ngmax};
                break;
            case NeighborhoodType::fullNeighborList:
                builder = cstone::ijloop::CpuFullNbListNeighborhoodBuilder{d.ngmax};
                break;
            default:
                throw std::runtime_error("only always-traverse and full-neighbor-list are available on CPUs");
                break;
        }

        std::visit(
            [&](auto const& nb) {
                neighborhood =
                    nb.build(d.treeView, box, d.size(), groups, d.x.data(), d.y.data(), d.z.data(), d.h.data());
            },
            builder);
    }

    template<class... Args>
    void ijLoop(Args&&... args) const
    {
        std::visit([&](auto const& nb) { nb.ijLoop(std::forward<Args>(args)...); }, neighborhood);
    }

private:
    std::variant<NeighborhoodDataType<cstone::ijloop::CpuAlwaysTraverseNeighborhoodBuilder>,
                 NeighborhoodDataType<cstone::ijloop::CpuFullNbListNeighborhoodBuilder>>
                     neighborhood;
    NeighborhoodType neighborhoodType = NeighborhoodType::alwaysTraverse;
};

} // namespace sph
