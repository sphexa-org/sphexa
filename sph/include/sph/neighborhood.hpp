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
    clusteredNeighborList
};

template<class Neighborhood>
using NeighborhoodDataType = decltype(std::declval<Neighborhood>().build(
    std::declval<cstone::OctreeNsView<sph::SphTypes::CoordinateType, sph::SphTypes::KeyType>>(),
    std::declval<cstone::Box<sph::SphTypes::CoordinateType>>(), 0, std::declval<cstone::GroupView>(),
    std::declval<sph::SphTypes::CoordinateType*>(), std::declval<sph::SphTypes::CoordinateType*>(),
    std::declval<sph::SphTypes::CoordinateType*>(), std::declval<sph::SphTypes::HydroType*>()));

template<class Neighborhood>
using NeighborhoodSubgroupType =
    decltype(std::declval<NeighborhoodDataType<Neighborhood>>().subgroup(std::declval<cstone::GroupView>()));

struct NeighborhoodData
{
    NeighborhoodData() {}

    void setType(NeighborhoodType type)
    {
        if (type == NeighborhoodType::clusteredNeighborList)
            throw std::invalid_argument("clustered neighbor lists are not available on CPU");
        neighborhoodType = type;
    }

    template<class Dataset, class T>
    void build(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box, bool /* subgroups */)
    {
        data.emplace<0>();

        std::variant<cstone::ijloop::CpuAlwaysTraverseNeighborhood, cstone::ijloop::CpuFullNbListNeighborhood>
            neighborhood;

        switch (neighborhoodType)
        {
            case NeighborhoodType::alwaysTraverse:
                neighborhood = cstone::ijloop::CpuAlwaysTraverseNeighborhood{d.ngmax};
                break;
            case NeighborhoodType::fullNeighborList:
                neighborhood = cstone::ijloop::CpuFullNbListNeighborhood{d.ngmax};
                break;
            case NeighborhoodType::clusteredNeighborList:
                throw std::runtime_error("clustered neighbor lists are not available on CPU");
                break;
        }

        std::visit(
            [&](auto const& nb)
            { data = nb.build(d.treeView, box, d.size(), groups, d.x.data(), d.y.data(), d.z.data(), d.h.data()); },
            neighborhood);
    }

    template<class... Args>
    void ijLoop(Args&&... args) const
    {
        std::visit([&](auto const& nb) { nb.ijLoop(std::forward<Args>(args)...); }, data);
    }

private:
    std::variant<NeighborhoodDataType<cstone::ijloop::CpuAlwaysTraverseNeighborhood>,
                 NeighborhoodDataType<cstone::ijloop::CpuFullNbListNeighborhood>>
                     data;
    NeighborhoodType neighborhoodType = NeighborhoodType::alwaysTraverse;
};

} // namespace sph
