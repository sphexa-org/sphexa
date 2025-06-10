/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Neighbor search on the GPU using fixed-size particle clusters similar to GROMACS.
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <tuple>
#include <type_traits>

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

#include "cstone/traversal/ijloop/gpu_superclusternblist/build.cuh"
#include "cstone/traversal/ijloop/gpu_superclusternblist/loop.cuh"
#include "cstone/traversal/ijloop/gpu_superclusternblist/temporaries.cuh"
#include "cstone/tree/octree.hpp"

namespace cstone::ijloop
{

namespace gpu_supercluster_nb_list_neighborhood_detail
{

template<class Config, class Tc, class Th>
struct GpuSuperclusterNbListNeighborhoodImpl
{
    Box<Tc> box               = {0, 0};
    LocalIndex firstValidBody = 0, totalBodies = 0, firstBody = 0, lastBody = 0;
    const Tc *x = nullptr, *y = nullptr, *z = nullptr;
    const Th* h = nullptr;
    util::UniqueDevicePtr<std::uint32_t[]> neighborData;
    util::UniqueDevicePtr<SuperclusterInfo[]> superclusterInfo;
    unsigned ncmax       = 0;
    std::size_t numBytes = 0;

    template<class... In, class... Out, class Interaction, class Postamble>
    void
    ijLoop(std::tuple<In*...> input, std::tuple<Out*...> output, Interaction&& interaction, Postamble&& postamble) const
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
        GpuSuperclusterNbListNeighborhoodImpl const& parent;
        GroupView groups;
        util::UniqueDevicePtr<typename Config::SuperclusterParticleMask[]> activeMasks;
        util::UniqueDevicePtr<SuperclusterInfo[]> superclusterInfo;
        LocalIndex numISuperclusters;

        template<class... In, class... Out, class Interaction, class Postamble>
        void ijLoop(std::tuple<In*...> input,
                    std::tuple<Out*...> output,
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

        auto activeMasks = util::deviceAlloc<typename Config::SuperclusterParticleMask[]>(numISuperclusters);
        checkGpuErrors(cudaMemsetAsync(activeMasks.get(), 0,
                                       sizeof(typename Config::SuperclusterParticleMask) * numISuperclusters));

        constexpr unsigned numThreads = 256;
        const unsigned numBlocks      = iceil(groups.numGroups, numThreads);
        computeActiveMasks<Config>
            <<<numBlocks, numThreads>>>(firstISupercluster, firstValidBody, groups, activeMasks.get());
        checkGpuErrors(cudaGetLastError());

        auto activeSuperclusterInfo = util::deviceAlloc<SuperclusterInfo[]>(numISuperclusters);

        SuperclusterInfo* lastCopied = thrust::copy_if(
            thrust::device, superclusterInfo.get(), superclusterInfo.get() + numISuperclusters,
            activeSuperclusterInfo.get(),
            [activeMasksPtr = activeMasks.get(), firstISupercluster] __device__(const SuperclusterInfo& info)
            { return activeMasksPtr[info.index - firstISupercluster] != 0; });
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

        auto [tmpOrOutput, tmpHolder] = allocateTemporaries<Config, Tc, Th>(
            firstBody, lastBody, makeConstRestrict(input), output, std::forward<Interaction>(interaction));

        if constexpr (Config::symmetric)
        {
            constexpr unsigned threads = 256;
            const unsigned numBlocks   = iceil(numBodies, threads);
            initResult<<<numBlocks, threads>>>(firstBody, lastBody, x, y, z, h, makeConstRestrict(input), tmpOrOutput,
                                               std::forward<Interaction>(interaction));
            checkGpuErrors(cudaGetLastError());
        }

        constexpr unsigned numSuperclustersPerBlock = 64 / (Config::iThreads * Config::jSize);
        const dim3 blockSize                        = {Config::iThreads, Config::jSize, numSuperclustersPerBlock};
        const unsigned numBlocks                    = iceil(numISuperclusters, numSuperclustersPerBlock);
        const unsigned sharedMem =
            numSuperclustersPerBlock *
            runIjLoopSharedMemPerSupercluster<Config, Tc, Th, std::decay_t<decltype(makeConstRestrict(input))>,
                                              std::decay_t<Interaction>>(ncmax);
        const auto runKernel = [&](auto usePbc)
        {
            runIjLoop<Config, numSuperclustersPerBlock, decltype(usePbc)::value><<<numBlocks, blockSize, sharedMem>>>(
                box, firstValidBody, totalBodies, firstBody, lastBody, x, y, z, h, makeConstRestrict(input),
                tmpOrOutput, std::forward<Interaction>(interaction), std::forward<Postamble>(postamble),
                neighborData.get(), superclusterInfo, activeMasks, ncmax);
            checkGpuErrors(cudaGetLastError());
        };
        if (box.boundaryX() == BoundaryType::periodic | box.boundaryY() == BoundaryType::periodic |
            box.boundaryZ() == BoundaryType::periodic)
            runKernel(std::true_type());
        else
            runKernel(std::false_type());

        if constexpr (Config::symmetric && !std::is_same<std::decay_t<Postamble>, detail::EmptyPostamble>())
        {
            constexpr unsigned threads = 256;
            const unsigned numBlocks   = iceil(totalBodies, threads);
            applyPostamble<<<numBlocks, threads>>>(firstBody, lastBody, firstValidBody, x, y, z, h,
                                                   makeConstRestrict(input), makeConstRestrict(tmpOrOutput), output,
                                                   std::forward<Postamble>(postamble));
            checkGpuErrors(cudaGetLastError());
            // device sync required due to possible use of allocated temporaries
            checkGpuErrors(cudaDeviceSynchronize());
        }
    }
};

template<unsigned ISize            = 8,
         unsigned JSize            = 8,
         unsigned SuperclusterSize = ISize * std::max(JSize, GpuConfig::warpSize / ISize),
         bool Compress             = false,
         bool Symmetric            = true>
struct GpuSuperclusterNbListNeighborhoodConfig
{
    static_assert((ISize & (ISize - 1)) == 0, "ISize must be power of two");
    static_assert((JSize & (JSize - 1)) == 0, "JSize must be power of two");
    static_assert(SuperclusterSize % ISize == 0, "SuperclusterSize must be divisible by ISize");
    static_assert(SuperclusterSize % JSize == 0, "SuperclusterSize must be divisible by JSize");

    static constexpr unsigned iSize            = ISize;
    static constexpr unsigned jSize            = JSize;
    static constexpr unsigned superclusterSize = SuperclusterSize;
    static constexpr bool compress             = Compress;
    static constexpr bool symmetric            = Symmetric;

    static constexpr unsigned iClustersPerSupercluster = superclusterSize / iSize;
    static constexpr unsigned iThreads                 = std::max(iSize, GpuConfig::warpSize / jSize);
    static constexpr unsigned numWarpsPerInteraction = (iSize * jSize + GpuConfig::warpSize - 1) / GpuConfig::warpSize;

    template<unsigned NewISize, unsigned NewJSize>
    using withClusterSize =
        GpuSuperclusterNbListNeighborhoodConfig<NewISize, NewJSize, SuperclusterSize, Compress, Symmetric>;
    template<unsigned NewSuperclusterSize>
    using withSuperclusterSize =
        GpuSuperclusterNbListNeighborhoodConfig<ISize, JSize, NewSuperclusterSize, Compress, Symmetric>;
    using withCompression = GpuSuperclusterNbListNeighborhoodConfig<ISize, JSize, SuperclusterSize, true, Symmetric>;
    using withoutCompression =
        GpuSuperclusterNbListNeighborhoodConfig<ISize, JSize, SuperclusterSize, false, Symmetric>;
    template<bool NewSymmetric>
    using setSymmetry = GpuSuperclusterNbListNeighborhoodConfig<ISize, JSize, SuperclusterSize, Compress, NewSymmetric>;

    // per-particle mask type for superclusters, always 32 or 64 bits to support atomic operations
    using SuperclusterParticleMask = std::conditional_t<(superclusterSize > 32), unsigned long long, unsigned>;
    static_assert(superclusterSize <= 64, "superclusters with more than 64 particles are not supported");
};

} // namespace gpu_supercluster_nb_list_neighborhood_detail

template<class Config = gpu_supercluster_nb_list_neighborhood_detail::GpuSuperclusterNbListNeighborhoodConfig<>>
struct GpuSuperclusterNbListNeighborhood
{
    template<unsigned ISize, unsigned JSize>
    using withClusterSize = GpuSuperclusterNbListNeighborhood<typename Config::template withClusterSize<ISize, JSize>>;
    template<unsigned SuperclusterSize>
    using withSuperclusterSize =
        GpuSuperclusterNbListNeighborhood<typename Config::template withSuperclusterSize<SuperclusterSize>>;
    using withCompression    = GpuSuperclusterNbListNeighborhood<typename Config::withCompression>;
    using withoutCompression = GpuSuperclusterNbListNeighborhood<typename Config::withoutCompression>;
    template<bool Symmetric>
    using setSymmetry     = GpuSuperclusterNbListNeighborhood<typename Config::template setSymmetry<Symmetric>>;
    using withSymmetry    = setSymmetry<true>;
    using withoutSymmetry = setSymmetry<false>;

    static constexpr unsigned iSize            = Config::iSize;
    static constexpr unsigned jSize            = Config::jSize;
    static constexpr unsigned superclusterSize = Config::superclusterSize;
    static constexpr bool compress             = Config::compress;
    static constexpr bool symmetric            = Config::symmetric;

    unsigned ncmax;
    std::size_t upperBoundBytesPerParticle = 128;

    template<class Tc, class KeyType, class Th>
    gpu_supercluster_nb_list_neighborhood_detail::GpuSuperclusterNbListNeighborhoodImpl<Config, Tc, Th>
    build(const OctreeNsView<Tc, KeyType>& tree,
          const Box<Tc>& box,
          LocalIndex totalBodies,
          GroupView groups,
          const Tc* x,
          const Tc* y,
          const Tc* z,
          const Th* h) const
    {
        using namespace gpu_supercluster_nb_list_neighborhood_detail;

        if (totalBodies == 0) return {};

        // align particle indices to cluster boundaries: insert invalid particles at the beginning of the particle
        // array, to make sure particle with index firstBody is the first particle of a supercluster
        const LocalIndex firstValidBody = clusterOffset<Config>(groups.firstBody);
        groups.firstBody += firstValidBody;
        groups.lastBody += firstValidBody;
        totalBodies += firstValidBody;

        assert(groups.firstBody % Config::superclusterSize == 0);

        // modify particle pointers to adhere to supercluster-aligned indexing
        x -= firstValidBody;
        y -= firstValidBody;
        z -= firstValidBody;
        h -= firstValidBody;

        const LocalIndex firstISupercluster = superclusterIndex<Config>(groups.firstBody);
        const LocalIndex lastISupercluster  = superclusterIndex<Config>(groups.lastBody - 1) + 1;
        const LocalIndex numISuperclusters  = lastISupercluster - firstISupercluster;

        if (numISuperclusters == 0) return {};

        // first main data array: a hugely oversized array to store neighbor indices is allocated in *virtual* memory,
        // as its final size is unknown a priori; in *physical* memory, only the required pages will be allocated
        std::size_t neighborDataVirtualSize = upperBoundBytesPerParticle * totalBodies / sizeof(std::uint32_t);
        auto neighborData                   = util::deviceAllocVirtual<std::uint32_t[]>(neighborDataVirtualSize);

        // second main data array: storing some data for each supercluster
        auto superclusterInfo = initSuperclusterInfo(firstISupercluster, numISuperclusters);

        // temporary data arrays, only used during build
        auto jClusterBboxes = computeJClusterBboxes<Config>(firstValidBody, totalBodies, x, y, z, h);
        auto nodeRMax       = computeNodeRMax<Config>(tree, h);
        auto superclusterSplitMasks =
            computeSuperclusterSplitMasks<Config>(firstValidBody, groups, firstISupercluster, numISuperclusters);

        // main build with octree traversal
        std::size_t neighborDataSize =
            buildNbList<Config>(tree, box, totalBodies, groups, x, y, z, h, firstValidBody, numISuperclusters,
                                jClusterBboxes.get(), nodeRMax.get(), ncmax, superclusterSplitMasks.get(),
                                neighborData.get(), neighborDataVirtualSize, superclusterInfo.get());

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
