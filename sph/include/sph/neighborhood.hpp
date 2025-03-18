#pragma once

#include <any>

#include "cstone/traversal/groups.hpp"
#include "cstone/traversal/ijloop/cpu.hpp"

namespace sph
{

namespace detail
{

template<class Dataset>
inline auto buildNeighborhoodImpl(const cstone::GroupView& groups, Dataset& d,
                                  const cstone::Box<typename Dataset::RealType>& box)
{
    return cstone::ijloop::CpuFullNbListNeighborhood{d.ngmax}.build(d.treeView, box, d.size(), groups, d.x.data(),
                                                                    d.y.data(), d.z.data(), d.h.data());
}

template<class Dataset>
using NeighborhoodType = decltype(buildNeighborhoodImpl(std::declval<cstone::GroupView>(), std::declval<Dataset&>(),
                                                        std::declval<cstone::Box<typename Dataset::RealType>>()));

} // namespace detail

template<class Dataset>
inline void buildNeighborhood(const cstone::GroupView& groups, Dataset& d,
                              const cstone::Box<typename Dataset::RealType>& box)
{
    d.neighborhood = detail::buildNeighborhoodImpl(groups, d, box);
}

template<class Dataset>
inline const detail::NeighborhoodType<Dataset>& getNeighborhood(const Dataset& d)
{
    return std::any_cast<const detail::NeighborhoodType<Dataset>&>(d.neighborhood);
}

} // namespace sph
