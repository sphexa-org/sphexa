#pragma once

#include <any>
#include <bit>
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

template<class Dataset, class NeighborhoodInfo>
using NeighborhoodData = decltype(std::declval<NeighborhoodInfo>().build(
    std::declval<Dataset>().treeView, std::declval<cstone::Box<typename Dataset::RealType>>(), 0,
    std::declval<cstone::GroupView>(), rawPtr(std::declval<Dataset>().devData.x),
    rawPtr(std::declval<Dataset>().devData.y), rawPtr(std::declval<Dataset>().devData.z),
    rawPtr(std::declval<Dataset>().devData.h)));

template<class Dataset, class NeighborhoodInfoVariant>
struct NeighborhoodDataVariant;

template<class Dataset, class... NeighborhoodInfos>
struct NeighborhoodDataVariant<Dataset, std::variant<NeighborhoodInfos...>>
{
    using type = std::variant<NeighborhoodData<Dataset, NeighborhoodInfos>...>;
};

template<unsigned NcMax, class Dataset, class... Variants>
void setInfo(Dataset const& d, std::variant<Variants...>& info)
{
    if (!std::holds_alternative<NeighborhoodInfo<NcMax>>(info) && NcMax == std::bit_ceil(d.ngmax) * 2)
        info = NeighborhoodInfo<NcMax>{};
}

template<class Dataset, class... Variants>
void setInfo(Dataset& d, std::variant<Variants...>& v)
{
    (..., setInfo<Variants::ncMax>(d, v));
}

template<class Dataset>
struct NeighborhoodDataGpu
{
    using InfoVariant =
        std::variant<NeighborhoodInfo<128>, NeighborhoodInfo<256>, NeighborhoodInfo<512>, NeighborhoodInfo<1024>>;
    using DataVariant = typename NeighborhoodDataVariant<Dataset, InfoVariant>::type;

    InfoVariant info;
    DataVariant data;

    NeighborhoodDataGpu() {}

    void build(const cstone::GroupView& groups, Dataset& d, const cstone::Box<typename Dataset::RealType>& box)
    {
        setInfo(d, info);
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
