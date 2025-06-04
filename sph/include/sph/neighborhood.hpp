#pragma once

#include "cstone/traversal/groups.hpp"
#include "cstone/traversal/ijloop/cpu_fullnblist.hpp"
#include "sph/types.hpp"

namespace sph
{

using CpuNeighborhood = cstone::ijloop::CpuFullNbListNeighborhood;

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

    template<class Dataset, class T>
    void build(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box, bool /* subgroups */)
    {
        data = CpuNeighborhood{d.ngmax}.build(d.treeView, box, d.size(), groups, d.x.data(), d.y.data(), d.z.data(),
                                              d.h.data());
    }

    template<class... Args>
    void ijLoop(Args&&... args) const
    {
        data.ijLoop(std::forward<Args>(args)...);
    }

private:
    NeighborhoodDataType<CpuNeighborhood> data;
};

} // namespace sph
