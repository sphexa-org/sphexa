#pragma once

#include <memory>
#include <variant>

#include "cstone/traversal/groups.hpp"
#include "sph/neighborhood.hpp"

#if defined(__CUDACC__) || defined(__HIP__)
#include "cstone/traversal/ijloop/gpu_alwaystraverse.cuh"
#include "cstone/traversal/ijloop/gpu_superclusternblist.cuh"
#include "neighborhood_gpu.hpp"
#endif

namespace sph
{

struct DeviceNeighborhoodData
{
    DeviceNeighborhoodData();
    ~DeviceNeighborhoodData();

    template<class Dataset, class T>
    void build(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box, bool clustered);

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
    void build(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box, bool clustered)
    {
        data.emplace<0>();

        std::variant<cstone::ijloop::GpuAlwaysTraverseNeighborhood, ClusteredNeighborhood<true>> neighborhood;
        const unsigned ncmax = std::bit_ceil(d.ngmax * 2);

        if (!clustered)
            neighborhood = cstone::ijloop::GpuAlwaysTraverseNeighborhood{d.ngmax};
        else
            neighborhood = ClusteredNeighborhood<true>{ncmax};

        std::visit(
            [&](auto const& nb)
            {
                data = nb.build(d.treeView, box, d.size(), groups, rawPtr(d.devData.x), rawPtr(d.devData.y),
                                rawPtr(d.devData.z), rawPtr(d.devData.h));
            },
            neighborhood);
    }

    template<class... Args>
    void ijLoop(Args&&... args) const
    {
        std::visit([&](auto const& nb) { nb.ijLoop(std::forward<Args>(args)...); }, data);
    }

    std::variant<NeighborhoodDataType<cstone::ijloop::GpuAlwaysTraverseNeighborhood>,
                 NeighborhoodDataType<ClusteredNeighborhood<true>>>
        data;
};

template<class Dataset, class T>
void DeviceNeighborhoodData::build(const cstone::GroupView& groups, Dataset& d, const cstone::Box<T>& box,
                                   bool clustered)
{
    assert(impl);
    impl->build(groups, d, box, clustered);
}

template<class... Args>
void DeviceNeighborhoodData::ijLoop(Args&&... args) const
{
    assert(impl);
    impl->ijLoop(std::forward<Args>(args)...);
}
#endif

} // namespace sph
