/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Neighbor search on the GPU using fixed-size particle clusters similar to GROMACS.
 * Publication about GROMACS' implementation: "A flexible algorithm for calculating pair interactions on SIMD
 * architectures" by Pall and Hess, 2013
 *
 * @author Felix Thaler <thaler@cscs.ch>
 *
 * The ij-loop implementation is close to GROMACS' one, but significantly more general. The neighborhood data structure
 * differs significantly and optionally incorporates index list compression. The build process is a custom development,
 * based on the cornerstone octree.
 *
 * In the present implementation, cluster sizes can be chosen at compile-time using template parameters. GROMACS uses
 * clusters of 8 particles on GPUs, which is also the default here. Then, 8 consecutive particles in memory are assumed
 * to be acluster. The following memory layout is assumed for particle data:
 *
 * 0           firstBody    lastBody    totalBodies
 * |-----------|------------|-----------|
 *   halo data   local data   halo data
 *
 * Halo data is only read if the halo particles are neighbors to local particles (with firstBody <= i < lastBody).
 * For efficient clustering however, the above memory layout is internally shifted to align firstBody with a cluster
 * boundary. Thus, the updated layout looks as follows (with the value of firstValidBody added to firstBody, lastBody,
 * and totalBodies):
 *
 * 0         firstValidBody  firstBody    lastBody    totalBodies
 * |---------|---------------|------------|-----------|
 *   padding     halo data     local data   halo data
 *
 * The particle-particle interactions in the ij-loop are implemented in terms of cluster-cluster interactions.
 * Thus 8x8 particle interactions are computed at once using two warps (in the most common case of warp size 32).
 * The following table summarizes how the 64 particle-particle interactions of one cluster-cluster interaction are
 * distributed among the available warps (w0, w1) and lanes (l0, ..., l31).
 *
 * | j \ i  | i0 + 0 | i0 + 1 | i0 + 2 | i0 + 3 | i0 + 4 | i0 + 5 | i0 + 6 | i0 + 7 |
 * |--------|--------|--------|--------|--------|--------|--------|--------|--------|
 * | j0 + 0 | w0/l0  | w0/l1  | w0/l2  | w0/l3  | w0/l4  | w0/l5  | w0/l6  | w0/l7  |
 * | j0 + 1 | w0/l8  | w0/l9  | w0/l10 | w0/l11 | w0/l12 | w0/l13 | w0/l14 | w0/l15 |
 * | j0 + 2 | w0/l16 | w0/l17 | w0/l18 | w0/l19 | w0/l20 | w0/l21 | w0/l22 | w0/l23 |
 * | j0 + 3 | w0/l24 | w0/l25 | w0/l26 | w0/l27 | w0/l28 | w0/l29 | w0/l30 | w0/l31 |
 * | j0 + 4 | w1/l0  | w1/l1  | w1/l2  | w1/l3  | w1/l4  | w1/l5  | w1/l6  | w1/l7  |
 * | j0 + 5 | w1/l8  | w1/l9  | w1/l10 | w1/l11 | w1/l12 | w1/l13 | w1/l14 | w1/l15 |
 * | j0 + 6 | w1/l16 | w1/l17 | w1/l18 | w1/l19 | w1/l20 | w1/l21 | w1/l22 | w1/l23 |
 * | j0 + 7 | w1/l24 | w1/l25 | w1/l26 | w1/l27 | w1/l28 | w1/l29 | w1/l30 | w1/l31 |
 *
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

#include "cstone/traversal/ijloop/gpu_superclusternblist/build.cuh"
#include "cstone/traversal/ijloop/gpu_superclusternblist/loop.cuh"
#include "cstone/traversal/ijloop/temporaries.cuh"
#include "cstone/tree/octree.hpp"

namespace cstone::ijloop
{

namespace gpu_supercluster_nb_list_neighborhood_detail
{

template<class Config, class Tc, class ThP>
struct GpuSuperclusterNbListNeighborhood
{
    Box<Tc> box               = {0, 0};
    LocalIndex firstValidBody = 0, totalBodies = 0, firstBody = 0, lastBody = 0;
    const Tc *x = nullptr, *y = nullptr, *z = nullptr;
    ThP h;
    util::UniqueDevicePtr<std::uint32_t[]> neighborData;
    util::UniqueDevicePtr<SuperclusterInfo[]> superclusterInfo;
    unsigned ncmax       = 0;
    std::size_t numBytes = 0;

    template<class... In, class... Out, class Interaction, class Postamble>
    void ijLoop(const std::tuple<In*...>& input,
                const std::tuple<Out*...>& output,
                Interaction&& interaction,
                Postamble&& postamble) const
    {
        if (totalBodies == 0) return;

        assert(firstBody < lastBody);
        const LocalIndex firstISupercluster = superclusterIndex<Config>(firstBody);
        const LocalIndex lastISupercluster  = superclusterIndex<Config>(lastBody - 1) + 1;
        const LocalIndex numISuperclusters  = lastISupercluster - firstISupercluster;

        ijLoop(input, output, std::forward<Interaction>(interaction), std::forward<Postamble>(postamble),
               superclusterInfo.get(), numISuperclusters);
    }

    Statistics stats() const { return {.numBodies = lastBody - firstBody, .numBytes = numBytes}; }

    struct Subgroup
    {
        GpuSuperclusterNbListNeighborhood const& parent;
        GroupView groups;
        util::UniqueDevicePtr<typename Config::SuperclusterParticleMask[]> activeMasks;
        util::UniqueDevicePtr<SuperclusterInfo[]> superclusterInfo;
        LocalIndex numISuperclusters;

        template<class... In, class... Out, class Interaction, class Postamble>
        void ijLoop(const std::tuple<In*...>& input,
                    const std::tuple<Out*...>& output,
                    Interaction&& interaction,
                    Postamble&& postamble) const
        {
            if (groups.numGroups == 0) return;

            parent.ijLoop(input, output, std::forward<Interaction>(interaction), std::forward<Postamble>(postamble),
                          superclusterInfo.get(), numISuperclusters, activeMasks.get());
        }
    };

    Subgroup subgroup(GroupView const& groups) const
    {
        static_assert(!Config::symmetric, "subgroup only supported in non-symmetric neighborhoods");
        const LocalIndex firstISupercluster = superclusterIndex<Config>(firstBody);
        const LocalIndex lastISupercluster  = superclusterIndex<Config>(lastBody - 1) + 1;
        const LocalIndex numISuperclusters  = lastISupercluster - firstISupercluster;

        auto activeMasks = computeActiveMasks<Config>(firstISupercluster, numISuperclusters, firstValidBody, groups);

        const auto superclusterIsActive =
            [activeMasksPtr = activeMasks.get(), firstISupercluster] __device__(const SuperclusterInfo& info)
        { return activeMasksPtr[info.index - firstISupercluster] != 0; };

        auto activeSuperclusterInfo = util::deviceAlloc<SuperclusterInfo[]>(numISuperclusters);
        SuperclusterInfo* lastCopied =
            thrust::copy_if(thrust::device, superclusterInfo.get(), superclusterInfo.get() + numISuperclusters,
                            activeSuperclusterInfo.get(), superclusterIsActive);
        const LocalIndex activeNumISuperclusters = lastCopied - activeSuperclusterInfo.get();

        return {*this, groups, std::move(activeMasks), std::move(activeSuperclusterInfo), activeNumISuperclusters};
    }

protected:
    template<class... In, class... Out, class Interaction, class Postamble, class Mask = void>
    void ijLoop(std::tuple<In*...> input,
                std::tuple<Out*...> output,
                Interaction&& interaction,
                Postamble&& postamble,
                const SuperclusterInfo* superclusterInfo,
                const LocalIndex numISuperclusters,
                const Mask* activeMasks = nullptr) const
    {
        const LocalIndex numBodies = lastBody - firstBody;
        if (numBodies == 0) return;

        // modify particle pointers to adhere to supercluster-aligned indexing
        util::for_each_tuple([&](auto& ptr) { ptr -= firstValidBody; }, input);
        util::for_each_tuple([&](auto& ptr) { ptr -= firstValidBody; }, output);

        // for symmetric neighborhoods where the reduction returns more values than the postamble, temporary arrays have
        // to be allocated; in all other cases, this functions just returns the output data pointers
        auto [tmpOrOutput, tmpHolder] = allocateTemporaries<Config, Tc, ThP>(
            firstBody, lastBody, makeConst(input), output, std::forward<Interaction>(interaction));

        if constexpr (Config::symmetric)
        {
            // in the symmetric case, the output arrays need to be initialized beforehand due to the unordered atomic
            // updates in the main loop
            initResult<Config>(firstBody, lastBody, x, y, z, h, makeConst(input), tmpOrOutput,
                               std::forward<Interaction>(interaction));
        }

        runIjLoop<Config>(box, firstValidBody, totalBodies, firstBody, lastBody, x, y, z, h, makeConst(input),
                          tmpOrOutput, std::forward<Interaction>(interaction), std::forward<Postamble>(postamble),
                          neighborData.get(), superclusterInfo, numISuperclusters, activeMasks);

        if constexpr (Config::symmetric)
        {
            // the postamble has to be applied in a separate step for symmetric neighborhoods
            applyPostamble<Config>(firstBody, lastBody, firstValidBody, x, y, z, h, makeConst(input),
                                   makeConst(tmpOrOutput), output, std::forward<Postamble>(postamble));

            // device sync required due to possible use of allocated temporaries
            checkGpuErrors(cudaDeviceSynchronize());
        }
    }
};

template<unsigned ISize            = 8,
         unsigned JSize            = 8,
         unsigned SuperclusterSize = ISize * std::max(JSize, GpuConfig::warpSize / ISize),
         class WarpCompression     = NibbleWarpCompression<false>,
         bool Symmetric            = true>
struct GpuSuperclusterNbListNeighborhoodConfig
{
    static_assert((ISize & (ISize - 1)) == 0, "ISize must be power of two");
    static_assert((JSize & (JSize - 1)) == 0, "JSize must be power of two");
    static_assert(SuperclusterSize % ISize == 0, "SuperclusterSize must be divisible by ISize");
    static_assert(SuperclusterSize % JSize == 0, "SuperclusterSize must be divisible by JSize");
    static_assert(ISize * JSize >= GpuConfig::warpSize, "ISize * JSize must be at least warpSize");
    static_assert(!WarpCompression::perThread, "Requires !WarpCompression::perThread");

    static constexpr unsigned iSize            = ISize;
    static constexpr unsigned jSize            = JSize;
    static constexpr unsigned superclusterSize = SuperclusterSize;
    using Compression                          = WarpCompression;
    static constexpr bool symmetric            = Symmetric;

    static constexpr unsigned iClustersPerSupercluster = superclusterSize / iSize;
    static constexpr unsigned numWarpsPerInteraction = (iSize * jSize + GpuConfig::warpSize - 1) / GpuConfig::warpSize;

    template<unsigned NewISize, unsigned NewJSize>
    using withClusterSize =
        GpuSuperclusterNbListNeighborhoodConfig<NewISize, NewJSize, SuperclusterSize, WarpCompression, Symmetric>;
    template<unsigned NewSuperclusterSize>
    using withSuperclusterSize =
        GpuSuperclusterNbListNeighborhoodConfig<ISize, JSize, NewSuperclusterSize, WarpCompression, Symmetric>;
    template<class NewCompression>
    using withCompression =
        GpuSuperclusterNbListNeighborhoodConfig<ISize, JSize, SuperclusterSize, NewCompression, Symmetric>;
    template<bool NewSymmetric>
    using setSymmetry =
        GpuSuperclusterNbListNeighborhoodConfig<ISize, JSize, SuperclusterSize, WarpCompression, NewSymmetric>;

    // per-particle mask type for superclusters, always 32 or 64 bits to support atomic operations
    using SuperclusterParticleMask = std::conditional_t<(superclusterSize > 32), unsigned long long, unsigned>;
    static_assert(superclusterSize <= 64, "superclusters with more than 64 particles are not supported");
};

} // namespace gpu_supercluster_nb_list_neighborhood_detail

template<class Config = gpu_supercluster_nb_list_neighborhood_detail::GpuSuperclusterNbListNeighborhoodConfig<>>
struct GpuSuperclusterNbListNeighborhoodBuilder
{
    template<unsigned ISize, unsigned JSize>
    using withClusterSize =
        GpuSuperclusterNbListNeighborhoodBuilder<typename Config::template withClusterSize<ISize, JSize>>;
    template<unsigned SuperclusterSize>
    using withSuperclusterSize =
        GpuSuperclusterNbListNeighborhoodBuilder<typename Config::template withSuperclusterSize<SuperclusterSize>>;
    template<class Compression = NibbleWarpCompression<false>>
    using withCompression =
        GpuSuperclusterNbListNeighborhoodBuilder<typename Config::template withCompression<Compression>>;
    using withoutCompression = GpuSuperclusterNbListNeighborhoodBuilder<
        typename Config::template withCompression<DummyWarpCompression<false>>>;
    template<bool Symmetric>
    using setSymmetry     = GpuSuperclusterNbListNeighborhoodBuilder<typename Config::template setSymmetry<Symmetric>>;
    using withSymmetry    = setSymmetry<true>;
    using withoutSymmetry = setSymmetry<false>;

    static constexpr unsigned iSize            = Config::iSize;
    static constexpr unsigned jSize            = Config::jSize;
    static constexpr unsigned superclusterSize = Config::superclusterSize;
    using Compression                          = Config::Compression;
    static constexpr bool symmetric            = Config::symmetric;

    unsigned ncmax;
    std::size_t upperBoundBytesPerParticle = 128;

    template<class Tc, class KeyType, class ThP>
    gpu_supercluster_nb_list_neighborhood_detail::GpuSuperclusterNbListNeighborhood<Config, Tc, ThP>
    build(const OctreeNsView<Tc, KeyType>& tree,
          const Box<Tc>& box,
          LocalIndex totalBodies,
          GroupView groups,
          const Tc* x,
          const Tc* y,
          const Tc* z,
          ThP h) const
    {
        using namespace gpu_supercluster_nb_list_neighborhood_detail;

        if (totalBodies == 0) return {};

        // align particle indices to cluster boundaries: insert invalid particles at the beginning of the particle
        // array, to make sure particle with index firstBody is the first particle of a supercluster
        const LocalIndex firstValidBody = clusterOffset<Config>(groups.firstBody);
        groups.firstBody += firstValidBody;
        groups.lastBody += firstValidBody;
        totalBodies += firstValidBody;
        assert(groups.firstBody <= groups.lastBody);
        assert(groups.lastBody <= totalBodies);

        assert(groups.firstBody % Config::superclusterSize == 0);

        // modify particle pointers to adhere to supercluster-aligned indexing
        x -= firstValidBody;
        y -= firstValidBody;
        z -= firstValidBody;
        if constexpr (std::is_pointer_v<ThP>) h -= firstValidBody;

        const LocalIndex firstISupercluster = superclusterIndex<Config>(groups.firstBody);
        const LocalIndex lastISupercluster  = superclusterIndex<Config>(groups.lastBody - 1) + 1;
        const LocalIndex numISuperclusters  = lastISupercluster - firstISupercluster;

        if (numISuperclusters == 0) return {};

        // first main data array: a hugely oversized array to store neighbor indices is allocated in *virtual* memory,
        // as its final size is unknown a priori; in *physical* memory, only the required pages will be allocated
        const std::size_t neighborDataVirtualSize = upperBoundBytesPerParticle * totalBodies / sizeof(std::uint32_t);
        auto neighborData                         = util::deviceAllocVirtual<std::uint32_t[]>(neighborDataVirtualSize);

        // second main data array: storing some data for each supercluster
        auto superclusterInfo = util::deviceAlloc<SuperclusterInfo[]>(numISuperclusters);

        // temporary data arrays, only used during build
        auto jClusterBboxes = computeJClusterBboxes<Config>(firstValidBody, totalBodies, x, y, z, h);

        using Th = std::remove_cvref_t<std::remove_pointer_t<ThP>>;
        util::UniqueDevicePtr<Th[]> nodeRMax;
        ThP nodeRMaxData;
        if constexpr (std::is_pointer_v<ThP>)
        {
            nodeRMax     = computeNodeRMax<Config>(tree, h + firstValidBody);
            nodeRMaxData = nodeRMax.get();
        }
        else { nodeRMaxData = h; }

        // main build with octree traversal
        std::size_t neighborDataSize = buildNbList<Config>(
            tree, box, totalBodies, groups, x, y, z, h, firstValidBody, numISuperclusters, jClusterBboxes.get(),
            nodeRMaxData, ncmax, neighborData.get(), neighborDataVirtualSize, superclusterInfo.get());

        // sort supercluster array by descending neighbor count for load balancing (schedule large work packages first)
        thrust::stable_sort(thrust::device, superclusterInfo.get(), superclusterInfo.get() + numISuperclusters);

        std::size_t numBytes = sizeof(std::uint32_t) * neighborDataSize + sizeof(SuperclusterInfo) * numISuperclusters;

        return {box,
                firstValidBody,
                totalBodies,
                groups.firstBody,
                groups.lastBody,
                x,
                y,
                z,
                h,
                std::move(neighborData),
                std::move(superclusterInfo),
                ncmax,
                numBytes};
    }
};

} // namespace cstone::ijloop
