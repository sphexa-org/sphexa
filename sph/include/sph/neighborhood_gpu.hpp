#pragma once

#include <memory>
#include <optional>
#include <variant>

#include "cstone/traversal/groups.hpp"
#include "sph/neighborhood.hpp"

#if defined(__CUDACC__) || defined(__HIP__)
#include "cstone/traversal/ijloop/gpu_superclusternblist.cuh"
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
template<bool Symmetric>
using ClusteredNeighborhood = cstone::ijloop::GpuSuperclusterNbListNeighborhood<>::withClusterSize<
    8, 8>::withSuperclusterSize<cstone::TravConfig::targetSize>::setSymmetry<Symmetric>::withCompression;

struct DeviceNeighborhoodData::Impl
{
    template<class Dataset, class T>
    void build(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box, bool subgroups)
    {
        if (subgroups && groups.firstBody == 0 && groups.lastBody == 0)
        {
            auto& nb = std::get<NeighborhoodDataType<ClusteredNeighborhood<false>>>(data);
            subgroupData.emplace(nb.subgroup(groups));
        }
        else
        {
            data.emplace<0>();
            subgroupData.reset();

            const unsigned ncmax = std::bit_ceil(d.ngmax * 2);

            std::variant<ClusteredNeighborhood<false>, ClusteredNeighborhood<true>> neighborhood;
            if (subgroups)
                neighborhood = ClusteredNeighborhood<false>{ncmax};
            else
                neighborhood = ClusteredNeighborhood<true>{ncmax};

            std::visit(
                [&](auto const& nb)
                {
                    data = nb.build(d.treeView, box, d.devData.size(), groups, rawPtr(d.devData.x), rawPtr(d.devData.y),
                                    rawPtr(d.devData.z), rawPtr(d.devData.h));
                },
                neighborhood);
        }
    }

    template<class... Args>
    void ijLoop(Args&&... args) const
    {
        if (subgroupData)
            subgroupData.value().ijLoop(std::forward<Args>(args)...);
        else
            std::visit([&](auto const& nb) { nb.ijLoop(std::forward<Args>(args)...); }, data);
    }

    std::variant<NeighborhoodDataType<ClusteredNeighborhood<false>>, NeighborhoodDataType<ClusteredNeighborhood<true>>>
                                                                          data;
    std::optional<NeighborhoodSubgroupType<ClusteredNeighborhood<false>>> subgroupData;
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
