#pragma once

#include <memory>
#include <optional>
#include <variant>

#include "cstone/traversal/groups.hpp"
#include "sph/neighborhood.hpp"

#if defined(__CUDACC__) || defined(__HIP__)
#include "cstone/traversal/ijloop/gpu_alwaystraverse.cuh"
#include "cstone/traversal/ijloop/gpu_superclusternblist.cuh"
#endif

namespace sph
{

struct DeviceNeighborhoodData
{
    DeviceNeighborhoodData();
    ~DeviceNeighborhoodData();

    void disableNeighborLists();

    template<class Dataset, class T>
    void build(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box, bool subgroups);

    template<class... Args>
    void ijLoop(cstone::ijloop::IjLoopData<Args...>) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

#if defined(__CUDACC__) || defined(__HIP__)
template<bool Symmetric>
using ClusteredNeighborhoodBuilder =
    cstone::ijloop::GpuSuperclusterNbListNeighborhoodBuilder<>::withClusterSize<8, cstone::GpuConfig::warpSize / 8>::
        withSuperclusterSize<cstone::GpuConfig::warpSize>::setSymmetry<Symmetric>::template withCompression<>;

struct DeviceNeighborhoodData::Impl
{
    template<class Dataset, class T>
    void build(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box, bool subgroups)
    {
        if (subgroups && groups.firstBody == 0 && groups.lastBody == 0)
        {
            subgroupNeighborhood.reset();
            std::visit(
                [&]<class Neighborhood>(Neighborhood const& nb)
                {
                    if constexpr (std::is_same_v<Neighborhood, NeighborhoodDataType<ClusteredNeighborhoodBuilder<true>,
                                                                                    cstone::execution::Gpu>>)
                        throw std::runtime_error("neighborhood does not support local time stepping");
                    else
                        subgroupNeighborhood.emplace(nb.subgroup(groups));
                },
                neighborhood);
        }
        else
        {
            neighborhood.emplace<0>();
            subgroupNeighborhood.reset();

            std::variant<cstone::ijloop::GpuAlwaysTraverseNeighborhoodBuilder, ClusteredNeighborhoodBuilder<false>,
                         ClusteredNeighborhoodBuilder<true>>
                builder;
            if (useNeighborLists)
            {

                if (subgroups)
                    builder = ClusteredNeighborhoodBuilder<false>{cstone::GpuConfig::warpSize * d.ngmax};
                else
                    builder = ClusteredNeighborhoodBuilder<true>{cstone::GpuConfig::warpSize * d.ngmax / 2};
            }
            else { builder = cstone::ijloop::GpuAlwaysTraverseNeighborhoodBuilder{d.ngmax}; }

            std::visit(
                [&](auto const& nb)
                {
                    neighborhood = nb.build(cstone::execution::gpuDefaultStream, d.treeView, box, d.size(), groups,
                                            rawPtr(d.x), rawPtr(d.y), rawPtr(d.z), rawPtr(d.h));
                },
                builder);
        }
    }

    template<class... Args>
    void ijLoop(cstone::ijloop::IjLoopData<Args...> ijData) const
    {
        const auto runIjLoop = [&](auto& nb) { nb.ijLoop(std::move(ijData)); };
        if (subgroupNeighborhood)
            std::visit(runIjLoop, subgroupNeighborhood.value());
        else
            std::visit(runIjLoop, neighborhood);
    }

    std::variant<NeighborhoodDataType<cstone::ijloop::GpuAlwaysTraverseNeighborhoodBuilder, cstone::execution::Gpu>,
                 NeighborhoodDataType<ClusteredNeighborhoodBuilder<false>, cstone::execution::Gpu>,
                 NeighborhoodDataType<ClusteredNeighborhoodBuilder<true>, cstone::execution::Gpu>>
        neighborhood;
    std::optional<std::variant<
        NeighborhoodSubgroupType<cstone::ijloop::GpuAlwaysTraverseNeighborhoodBuilder, cstone::execution::Gpu>,
        NeighborhoodSubgroupType<ClusteredNeighborhoodBuilder<false>, cstone::execution::Gpu>>>
         subgroupNeighborhood;
    bool useNeighborLists = true;
};

template<class Dataset, class T>
void DeviceNeighborhoodData::build(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box,
                                   bool subgroups)
{
    assert(impl);
    impl->build(groups, d, box, subgroups);
}

template<class... Args>
void DeviceNeighborhoodData::ijLoop(cstone::ijloop::IjLoopData<Args...> ijData) const
{
    assert(impl);
    impl->ijLoop(ijData);
}
#endif

} // namespace sph
