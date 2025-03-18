#pragma once

#include <any>
#include <cassert>
#include <stdexcept>
#include <variant>

#include "cstone/traversal/groups.hpp"
#include "cstone/traversal/ijloop/gpu_superclusternblist.cuh"

namespace sph
{

namespace detail
{

template<unsigned NcMax>
using NeighborhoodInfo = cstone::ijloop::GpuSuperclusterNbListNeighborhood<>::withClusterSize<
    8, 8>::withSuperclusterSize<cstone::TravConfig::targetSize>::withNcMax<NcMax>::withoutSymmetry::withCompression;

template<unsigned NcMax, class Dataset>
using NeighborhoodData = decltype(NeighborhoodInfo<NcMax>{}.build(
    std::declval<Dataset>().treeView, std::declval<cstone::Box<typename Dataset::RealType>>(), 0,
    std::declval<cstone::GroupView>(), rawPtr(std::declval<Dataset>().devData.x),
    rawPtr(std::declval<Dataset>().devData.y), rawPtr(std::declval<Dataset>().devData.z),
    rawPtr(std::declval<Dataset>().devData.h)));

template<class Dataset>
struct NeighborhoodDataGpu
{
    std::variant<NeighborhoodInfo<128>, NeighborhoodInfo<256>, NeighborhoodInfo<512>, NeighborhoodInfo<1024>> info;
    std::variant<NeighborhoodData<128, Dataset>, NeighborhoodData<256, Dataset>, NeighborhoodData<512, Dataset>,
                 NeighborhoodData<1024, Dataset>>
        data;

    NeighborhoodDataGpu() {}

    unsigned ncMax(const Dataset& d) const
    {
        if (d.ngmax <= 64) return 128;
        if (d.ngmax <= 128) return 256;
        if (d.ngmax <= 256) return 512;
        if (d.ngmax <= 512) return 1024;
        throw std::runtime_error("Maximum number of neighbors, ngmax, is too large");
    }

    template<unsigned NcMax>
    void setInfo()
    {
        if (!std::holds_alternative<NeighborhoodInfo<NcMax>>(info)) info = NeighborhoodInfo<NcMax>{};
    }

    void build(const cstone::GroupView& groups, Dataset& d, const cstone::Box<typename Dataset::RealType>& box)
    {
        switch (ncMax(d))
        {
            case 128: setInfo<128>(); break;
            case 256: setInfo<256>(); break;
            case 512: setInfo<512>(); break;
            case 1024: setInfo<1024>(); break;
            default: assert(false);
        }
        std::visit(
            [&](auto const& nb)
            {
                data = nb.build(d.treeView, box, d.size(), groups, rawPtr(d.devData.x), rawPtr(d.devData.y),
                                rawPtr(d.devData.z), rawPtr(d.devData.h));
            },
            info);
    }

    template<class... Args>
    void ijLoop(Args&&... args) const
    {
        std::visit([&](auto const& nb) { nb.ijLoop(std::forward<Args>(args)...); }, data);
    }
};

} // namespace detail

template<class Dataset>
inline void buildNeighborhoodGpu(const cstone::GroupView& groups, Dataset& d,
                                 const cstone::Box<typename Dataset::RealType>& box)
{
    if (!d.neighborhood.has_value()) d.neighborhood = detail::NeighborhoodDataGpu<Dataset>{};

    auto& nb = std::any_cast<detail::NeighborhoodDataGpu<Dataset>&>(d.neighborhood);
    nb.build(groups, d, box);
}

template<class Dataset>
inline const detail::NeighborhoodDataGpu<Dataset>& getNeighborhoodGpu(const Dataset& d)
{
    return std::any_cast<const detail::NeighborhoodDataGpu<Dataset>&>(d.neighborhood);
}

} // namespace sph
