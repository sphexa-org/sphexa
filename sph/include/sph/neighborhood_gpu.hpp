#pragma once

#include "cstone/traversal/groups.hpp"
#include "cstone/traversal/ijloop/gpu_alwaystraverse.cuh"

namespace sph
{

template<class Dataset>
inline auto buildNeighborhoodGpu(const cstone::GroupView& groups, Dataset& d,
                                 const cstone::Box<typename Dataset::RealType>& box)
{
    return cstone::ijloop::GpuAlwaysTraverseNeighborhood{d.ngmax}.build(d.treeView, box, d.size(), groups, d.x.data(),
                                                                        d.y.data(), d.z.data(), d.h.data());
}

template<class Dataset>
using NeighborhoodTypeGpu = decltype(buildNeighborhoodGpu(std::declval<cstone::GroupView>(), std::declval<Dataset&>(),
                                                          std::declval<cstone::Box<typename Dataset::RealType>>()));

} // namespace sph
