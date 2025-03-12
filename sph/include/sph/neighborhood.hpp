#pragma once

#include <any>

#include "cstone/traversal/groups.hpp"
#include "cstone/traversal/ijloop/cpu.hpp"

namespace sph
{

template<class Dataset>
inline auto buildNeighborhood(const cstone::GroupView& groups, Dataset& d,
                              const cstone::Box<typename Dataset::RealType>& box)
{
    return cstone::ijloop::CpuFullNbListNeighborhood{d.ngmax}.build(d.treeView, box, d.size(), groups, d.x.data(),
                                                                    d.y.data(), d.z.data(), d.h.data());
}

template<class Dataset>
using NeighborhoodType = decltype(buildNeighborhood(std::declval<cstone::GroupView>(), std::declval<Dataset&>(),
                                                    std::declval<cstone::Box<typename Dataset::RealType>>()));

template<class Dataset>
inline const NeighborhoodType<Dataset>& getNeighborhood(const Dataset& d)
{
    return std::any_cast<const NeighborhoodType<Dataset>&>(d.neighborhood);
}

} // namespace sph
