#pragma once

#include <any>
#include <bit>
#include <cassert>
#include <stdexcept>
#include <variant>

#include "cstone/traversal/groups.hpp"
#include "cstone/traversal/ijloop/gpu_alwaystraverse.cuh"
#include "cstone/traversal/ijloop/gpu_superclusternblist.cuh"

namespace sph
{

namespace detail
{

// TODO: re-enable symmetry once NB list build times are optimized for varying smoothing lengths
template<unsigned NcMax, bool Symmetric>
using NeighborhoodInfo =
    cstone::ijloop::GpuSuperclusterNbListNeighborhood<>::withClusterSize<8, 8>::withSuperclusterSize<
        cstone::TravConfig::targetSize>::withNcMax<NcMax>::template setSymmetry<false>::withCompression;

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

template<class Variant, class Dataset, class... Variants>
void setInfo(Dataset const& d, std::variant<Variants...>& info, bool alwaysTraverse)
{
    const unsigned ngmaxClustered = std::max(std::bit_ceil(d.ngmax * 2), 128u);
    if constexpr (std::is_same_v<Variant, cstone::ijloop::GpuAlwaysTraverseNeighborhood>)
    {
        if (alwaysTraverse || ngmaxClustered > 1024) info = Variant{d.ngmax};
    }
    else
    {
        if (!alwaysTraverse && !std::holds_alternative<Variant>(info) && Variant::ncMax == ngmaxClustered)
            info = Variant{};
    }
}

template<class Dataset, class... Variants>
void setInfo(Dataset& d, std::variant<Variants...>& v, bool alwaysTraverse)
{
    (..., setInfo<Variants>(d, v, alwaysTraverse));
}

template<class Dataset, bool Symmetric>
struct NeighborhoodDataGpu
{
    using InfoVariant = std::variant<cstone::ijloop::GpuAlwaysTraverseNeighborhood, NeighborhoodInfo<128, Symmetric>,
                                     NeighborhoodInfo<256, Symmetric>, NeighborhoodInfo<512, Symmetric>,
                                     NeighborhoodInfo<1024, Symmetric>>;
    using DataVariant = typename NeighborhoodDataVariant<Dataset, InfoVariant>::type;

    bool        alwaysTraverse;
    InfoVariant info;
    DataVariant data;

    void build(const cstone::GroupView& groups, Dataset& d, const cstone::Box<typename Dataset::RealType>& box)
    {
        setInfo(d, info, alwaysTraverse);
        data.template emplace<0>();
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

template<bool Symmetric = false, class Dataset>
inline void buildNeighborhoodGpu(const cstone::GroupView& groups, Dataset& d,
                                 const cstone::Box<typename Dataset::RealType>& box, bool clustered)
{
    if (!d.neighborhood.has_value()) d.neighborhood = detail::NeighborhoodDataGpu<Dataset, Symmetric>{};

    auto& nb          = std::any_cast<detail::NeighborhoodDataGpu<Dataset, Symmetric>&>(d.neighborhood);
    nb.alwaysTraverse = !clustered;
    nb.build(groups, d, box);
}

template<bool Symmetric = false, class Dataset>
inline const detail::NeighborhoodDataGpu<Dataset, Symmetric>& getNeighborhoodGpu(const Dataset& d)
{
    return std::any_cast<const detail::NeighborhoodDataGpu<Dataset, Symmetric>&>(d.neighborhood);
}

} // namespace sph
