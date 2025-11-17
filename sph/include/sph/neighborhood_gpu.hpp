#pragma once

#include <memory>
#include <optional>
#include <variant>

#include "cstone/traversal/groups.hpp"
#include "sph/neighborhood.hpp"

#if defined(__CUDACC__) || defined(__HIP__)
#include "cstone/traversal/ijloop/gpu_alwaystraverse.cuh"
#endif

namespace sph
{

struct DeviceNeighborhoodData
{
    DeviceNeighborhoodData();
    ~DeviceNeighborhoodData();

    template<class Dataset, class T>
    void build(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box, bool subgroups);

    template<class... Args>
    void ijLoop(Args&&... args) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

#if defined(__CUDACC__) || defined(__HIP__)
struct DeviceNeighborhoodData::Impl
{
    template<class Dataset, class T>
    void build(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box, bool subgroups)
    {
        if (subgroups && groups.firstBody == 0 && groups.lastBody == 0)
        {
            subgroupNeighborhood.reset();
            subgroupNeighborhood.emplace(neighborhood.subgroup(groups));
        }
        else
        {
            neighborhood = {};
            subgroupNeighborhood.reset();

            auto builder = cstone::ijloop::GpuAlwaysTraverseNeighborhoodBuilder{d.ngmax};

            neighborhood =
                builder.build(d.treeView, box, d.size(), groups, rawPtr(d.x), rawPtr(d.y), rawPtr(d.z), rawPtr(d.h));
        }
    }

    template<class... Args>
    void ijLoop(Args&&... args) const
    {
        const auto runIjLoop = [&](auto const& nb) { nb.ijLoop(std::forward<Args>(args)...); };
        if (subgroupNeighborhood)
            runIjLoop(subgroupNeighborhood.value());
        else
            runIjLoop(neighborhood);
    }

    NeighborhoodDataType<cstone::ijloop::GpuAlwaysTraverseNeighborhoodBuilder>                    neighborhood;
    std::optional<NeighborhoodSubgroupType<cstone::ijloop::GpuAlwaysTraverseNeighborhoodBuilder>> subgroupNeighborhood;
};

template<class Dataset, class T>
void DeviceNeighborhoodData::build(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box,
                                   bool subgroups)
{
    assert(impl);
    impl->build(groups, d, box, subgroups);
}

template<class... Args>
void DeviceNeighborhoodData::ijLoop(Args&&... args) const
{
    assert(impl);
    impl->ijLoop(std::forward<Args>(args)...);
}
#endif

} // namespace sph
