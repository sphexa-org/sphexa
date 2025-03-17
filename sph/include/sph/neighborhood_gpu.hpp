#pragma once

#include <any>

#include "cstone/traversal/groups.hpp"
#include "cstone/traversal/ijloop/gpu_superclusternblist.cuh"

namespace sph
{

template<class Dataset>
inline auto buildNeighborhoodGpu(const cstone::GroupView& groups, Dataset& d,
                                 const cstone::Box<typename Dataset::RealType>& box)
{
    using namespace cstone;
    using Neighborhood = ijloop::GpuSuperclusterNbListNeighborhood<>::withClusterSize<8, 8>::withSuperclusterSize<
        TravConfig::targetSize>::withNcMax<1024>::withoutSymmetry;

    if (!d.neighborhoodInfo.has_value()) d.neighborhoodInfo = Neighborhood{};

    const auto& nbInfo = std::any_cast<const Neighborhood&>(d.neighborhoodInfo);

    return nbInfo.build(d.treeView, box, d.size(), groups, rawPtr(d.devData.x), rawPtr(d.devData.y),
                        rawPtr(d.devData.z), rawPtr(d.devData.h));
}

template<class Dataset>
using NeighborhoodTypeGpu = decltype(buildNeighborhoodGpu(std::declval<cstone::GroupView>(), std::declval<Dataset&>(),
                                                          std::declval<cstone::Box<typename Dataset::RealType>>()));

template<class Dataset>
inline const NeighborhoodTypeGpu<Dataset>& getNeighborhoodGpu(const Dataset& d)
{
    return std::any_cast<const NeighborhoodTypeGpu<Dataset>&>(d.neighborhood);
}

} // namespace sph
