/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Neighbor search on GPU using persistent per-warp traversal and warp compression
 *
 * Each warp persistently fetches groups via an atomic counter, performs a single
 * warp-cooperative DFS traversal (singleTraversal) to collect raw neighbor indices
 * into a global memory scratch buffer, then compresses them into the final output.
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <tuple>

#include "cstone/cuda/memory.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/primitives/warpscan.cuh"
#include "cstone/traversal/boxoverlap.hpp"
#include "cstone/traversal/traversal.hpp"
#include "cstone/traversal/ijloop/atomic_update_ptr.cuh"
#include "cstone/traversal/ijloop/common.hpp"
#include "cstone/traversal/ijloop/compressneighbors.cuh"
#include "cstone/traversal/groups.hpp"
#include "cstone/traversal/ijloop/upsweep.cuh"
#include "cstone/traversal/ijloop/symmetric_loop.cuh"
#include "cstone/traversal/ijloop/temporaries.cuh"
#include "cstone/tree/octree.hpp"
#include "cstone/util/tuple_util.hpp"

namespace cstone::ijloop
{

namespace gpu_compressed_nb_list_neighborhood_detail
{

template<class WarpCompression = NibbleWarpCompression<true>, bool Symmetric = false>
struct GpuCompressedNbListNeighborhoodConfig
{
    static_assert(WarpCompression::perThread, "Requires WarpCompression::perThread == true");
    using Compression               = WarpCompression;
    static constexpr bool symmetric = Symmetric;
    template<class NewCompression>
    using withCompression    = GpuCompressedNbListNeighborhoodConfig<NewCompression, Symmetric>;
    using withoutCompression = GpuCompressedNbListNeighborhoodConfig<DummyWarpCompression<true>, Symmetric>;
    template<bool NewSymmetric>
    using setSymmetry     = GpuCompressedNbListNeighborhoodConfig<WarpCompression, NewSymmetric>;
    using withSymmetry    = setSymmetry<true>;
    using withoutSymmetry = setSymmetry<false>;
};

enum struct BuildStatus
{
    success = 0,
    neighbor_list_overflow,
    neighbor_data_overflow,
};

struct GlobalBuildData
{
    //! @brief total size of neighbor data, atomically increased during build to "allocate" required storage for each
    //! warp-sized group during build in a pre-allocated array
    unsigned long long neighborDataSize;
    //! @brief global group index counter, atomically incremented during build
    unsigned index;
    BuildStatus status;
    //! @brief global maximum number of neighbors
    unsigned maxNeighbors;
};

/*! decide if a neighbor index should be included in the symmetric neighbor list
 *
 * @param[in] i     own index
 * @param[in] j     neighbor index
 * @param[in] first start index of traversed entities
 * @param[in] last  end index of traversed entities
 *
 * @return true if the neighbor j should be included in the neighbor list of i, else false
 */
constexpr __forceinline__ bool includeNbSymmetric(unsigned i, unsigned j, unsigned first, unsigned last)
{
    // larger blockSize leads to more consecutive neighbors in list and thus improved neighbor list compression ratio
    // and cache locality
    constexpr unsigned blockSize = 1024;
    const bool s                 = ((i - first) / blockSize) % 2 == ((j - first) / blockSize) % 2;
    return (j < first) | (j >= last) | (i == j) | (i < j ? s : !s);
}

template<class Config, unsigned WarpsPerBlock, class Tc, class ThP, class KeyType>
__global__ __launch_bounds__(GpuConfig::warpSize* WarpsPerBlock) void gpuCompressedNbListNeighborhoodBuild(
    const OctreeNsView<Tc, KeyType> __grid_constant__ tree,
    const Box<Tc> __grid_constant__ box,
    const LocalIndex firstBody,
    const LocalIndex lastBody,
    const unsigned numGroups,
    const unsigned ngmax,
    const Tc* __restrict__ x,
    const Tc* __restrict__ y,
    const Tc* __restrict__ z,
    const ThP h,
    const std::remove_cvref_t<std::remove_pointer_t<ThP>>* nodeRMax,
    std::uint32_t* __restrict__ globalPool,
    std::uint32_t* __restrict__ neighborData,
    std::size_t* __restrict__ groupDataIndex,
    GlobalBuildData* __restrict__ globalBuildData,
    const std::size_t maxNeighborDataSize)
{
    using Th = std::remove_cvref_t<std::remove_pointer_t<ThP>>;

    const unsigned laneIdx     = threadIdx.x;
    const unsigned warpIdxGrid = WarpsPerBlock * blockIdx.x + threadIdx.y;

    std::uint32_t* const warpPool              = globalPool + warpIdxGrid * 2u * GpuConfig::warpSize * ngmax;
    std::uint32_t* const uncompressedNeighbors = warpPool;
    std::uint32_t* const compressedNeighbors   = warpPool + GpuConfig::warpSize * ngmax;

    unsigned warpMaxNc = 0;

    while (true)
    {
        unsigned groupIdx = 0;
        if (laneIdx == 0) groupIdx = atomicAdd(&globalBuildData->index, 1);
        groupIdx = shflSync(groupIdx, 0);
        if (groupIdx >= numGroups) break;

        const LocalIndex bodyBegin = firstBody + groupIdx * GpuConfig::warpSize;
        const LocalIndex bodyEnd   = std::min(bodyBegin + GpuConfig::warpSize, lastBody);
        const LocalIndex i         = bodyBegin + laneIdx;
        const bool active          = (i < bodyEnd);

        const Vec3<Tc> iPos = active ? Vec3<Tc>{x[i], y[i], z[i]} : Vec3<Tc>{0, 0, 0};

        Th hi = 0;
        if (active) hi = loadAtIndexIfPtr(h, i);

        const Th iRadius = Th(2) * hi * tree.searchExtFactor;

        constexpr auto pbc = BoundaryType::periodic;
        const bool anyPbc  = box.boundaryX() == pbc || box.boundaryY() == pbc || box.boundaryZ() == pbc;
        Th pbcCheckRadius  = iRadius;
        if constexpr (Config::symmetric && std::is_pointer_v<ThP>)
            pbcCheckRadius = std::max(iRadius, nodeRMax[0] * tree.searchExtFactor);
        const bool usePbc = anySync(
            active && (anyPbc && !insideBox(iPos, Vec3<Tc>{pbcCheckRadius, pbcCheckRadius, pbcCheckRadius}, box)));

        const auto continuationCriterion = [&](TreeNodeIndex idx) -> bool
        {
            const Vec3<Tc> srcCenter = tree.centers[idx];
            const Vec3<Tc> srcSize   = tree.sizes[idx];
            Th srcRadius             = 0;
            if constexpr (Config::symmetric && std::is_pointer_v<ThP>) srcRadius = nodeRMax[idx] * tree.searchExtFactor;
            const Th maxRadius = std::max(iRadius, srcRadius);
            const auto distance =
                usePbc ? minDistance(iPos, srcCenter, srcSize, box) : minDistance(iPos, srcCenter, srcSize);
            return anySync(active && norm2(distance) < maxRadius * maxRadius);
        };

        unsigned nc = 0;

        const auto endpointAction = [&](TreeNodeIndex idx)
        {
            if (!active) return;

            const TreeNodeIndex leafIdx = tree.internalToLeaf[idx];
            const LocalIndex leafBegin  = tree.layout[leafIdx];
            const LocalIndex leafEnd    = tree.layout[leafIdx + 1];

            for (LocalIndex j = leafBegin; j < leafEnd; ++j)
            {
                if (i == j || (Config::symmetric && !includeNbSymmetric(i, j, firstBody, lastBody))) continue;

                const Vec3<Tc> jPos = {x[j], y[j], z[j]};
                const Th jRadius    = Config::symmetric ? 2 * loadAtIndexIfPtr(h, j) * tree.searchExtFactor : Th(0);
                const Th maxRadius  = std::max(iRadius, jRadius);

                const bool isNeighbor =
                    (usePbc ? distanceSq<true>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box)
                            : distanceSq<false>(jPos[0], jPos[1], jPos[2], iPos[0], iPos[1], iPos[2], box)) <
                    maxRadius * maxRadius;
                if (isNeighbor)
                {
                    if (nc < ngmax) uncompressedNeighbors[nc * GpuConfig::warpSize + laneIdx] = j;
                    ++nc;
                }
            }
        };

        singleTraversal(tree.childOffsets, tree.parents, continuationCriterion, endpointAction);

        const unsigned groupMaxNc = warpMax(nc);
        warpMaxNc                 = std::max(warpMaxNc, groupMaxNc);
        if (groupMaxNc > ngmax && laneIdx == 0) globalBuildData->status = BuildStatus::neighbor_list_overflow;

        unsigned compressedSize = GpuConfig::warpSize;
        {
            typename Config::Compression comp(compressedNeighbors);
            for (unsigned nb = 0; nb < groupMaxNc; ++nb)
            {
                const unsigned neighbor = nb < nc ? uncompressedNeighbors[nb * GpuConfig::warpSize + laneIdx] : 0u;
                comp.add(neighbor, nb < nc);
            }
            compressedSize += (comp.numBytes() + sizeof(std::uint32_t) - 1) / sizeof(std::uint32_t);
        }

        unsigned long long dataStart = 0;
        if (laneIdx == 0) dataStart = atomicAdd(&globalBuildData->neighborDataSize, compressedSize);
        dataStart = shflSync(dataStart, 0);

        if (dataStart + compressedSize > maxNeighborDataSize)
        {
            if (laneIdx == 0) globalBuildData->status = BuildStatus::neighbor_data_overflow;
            break;
        }

        if (laneIdx == 0) groupDataIndex[groupIdx] = dataStart;

        neighborData[dataStart + laneIdx] = nc;

        for (unsigned idx = laneIdx; idx < compressedSize - GpuConfig::warpSize; idx += GpuConfig::warpSize)
            neighborData[dataStart + GpuConfig::warpSize + idx] = compressedNeighbors[idx];
    }

    if (laneIdx == 0) atomicMax(&globalBuildData->maxNeighbors, warpMaxNc);
}

template<class Config,
         unsigned WarpsPerBlock,
         class Tc,
         class ThP,
         class In,
         class Out,
         class Interaction,
         class Postamble>
__global__
__launch_bounds__(GpuConfig::warpSize* WarpsPerBlock) void runIjLoop(const Box<Tc> __grid_constant__ box,
                                                                     const LocalIndex firstBody,
                                                                     const LocalIndex lastBody,
                                                                     const Tc* __restrict__ x,
                                                                     const Tc* __restrict__ y,
                                                                     const Tc* __restrict__ z,
                                                                     const ThP h,
                                                                     const In input,
                                                                     const Out output,
                                                                     const Interaction interaction,
                                                                     const Postamble postamble,
                                                                     const std::uint32_t* __restrict__ neighborData,
                                                                     const std::size_t* __restrict__ groupDataIndex)
{
    assert(blockDim.x == GpuConfig::warpSize && blockDim.y == WarpsPerBlock && blockDim.z == 1);
    const unsigned laneIdx = threadIdx.x;
    const unsigned g       = WarpsPerBlock * blockIdx.x + threadIdx.y;

    const unsigned numGroups = iceil(lastBody - firstBody, GpuConfig::warpSize);
    if (g >= numGroups) return;

    const LocalIndex i = firstBody + g * GpuConfig::warpSize + laneIdx;

    const std::size_t dataStart = groupDataIndex[g];
    const unsigned nc           = i < lastBody ? neighborData[dataStart + laneIdx] : 0u;

    const auto iData = i < lastBody ? loadParticleData(x, y, z, h, input, i) : dummyParticleData(x, y, z, h, input, i);
    constexpr auto pbc = BoundaryType::periodic;
    const bool usePbc  = Config::symmetric
                             ? (box.boundaryX() == pbc) || (box.boundaryY() == pbc) || (box.boundaryZ() == pbc)
                             : requiresPbcHandling(box, iData);

    auto result = interaction(iData, iData, Vec3<Tc>{0, 0, 0}, Tc(0));

    const unsigned maxNc = warpMax(nc);
    if (maxNc > 0)
    {
        const auto iRadiusSq = radiusSq(iData);
        typename Config::Compression::Decompression decomp(neighborData + dataStart + GpuConfig::warpSize, nc);
        for (unsigned nb = 0; nb < maxNc; ++nb)
        {
            const unsigned j = decomp.next();

            if (nb < nc && i < lastBody)
            {
                const auto jData               = loadParticleData(x, y, z, h, input, j);
                const auto [ijPosDiff, distSq] = posDiffAndDistSq(usePbc, box, iData, jData);

                const bool iClose = distSq < iRadiusSq;
                const bool jClose = Config::symmetric && (std::is_pointer_v<ThP> ? distSq < radiusSq(jData) : iClose);

                if (iClose | jClose)
                {
                    const auto ijInteraction = interaction(iData, jData, ijPosDiff, distSq);
                    if (iClose) updateResult(result, ijInteraction);
                    if (jClose & (j >= firstBody) & (j < lastBody))
                    {
                        const auto jiInteraction =
                            selectSymmetric(ijInteraction, interaction(jData, iData, -ijPosDiff, distSq));
                        if constexpr (Config::symmetric)
                        {
                            util::for_each_tuple([j](auto* ptr, auto const& v) { atomicUpdatePtr(&ptr[j], v); }, output,
                                                 jiInteraction);
                        }
                    }
                }
            }
        }
    }

    if (i < lastBody)
    {
        if constexpr (Config::symmetric)
        {
            util::for_each_tuple([i](auto* ptr, auto const& v) { atomicUpdatePtr(&ptr[i], v); }, output, result);
        }
        else { storeParticleData(output, i, postamble(iData, unwrapModifiers(result))); }
    }
}

template<class Config, class Tc, class ThP>
struct GpuCompressedNbListNeighborhood
{
    Box<Tc> box          = {0, 0};
    LocalIndex firstBody = 0, lastBody = 0;
    const Tc *x = nullptr, *y = nullptr, *z = nullptr;
    ThP h = {};

    util::UniqueDevicePtr<std::uint32_t[]> neighborData;
    util::UniqueDevicePtr<std::size_t[]> groupDataIndex;
    std::size_t numBytesUsed = 0;

    template<class... In, class... Out, class Interaction, class Postamble>
    void ijLoop(std::tuple<In*...> const& input,
                std::tuple<Out*...> const& output,
                Interaction&& interaction,
                Postamble&& postamble) const
    {
        if (firstBody >= lastBody) return;

        const unsigned numGroups = iceil(lastBody - firstBody, GpuConfig::warpSize);

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

        constexpr unsigned warpsPerBlock = 4;
        const unsigned numBlocks         = iceil(numGroups, warpsPerBlock);
        constexpr dim3 blockSize         = {GpuConfig::warpSize, warpsPerBlock, 1};
        runIjLoop<Config, warpsPerBlock><<<numBlocks, blockSize>>>(
            box, firstBody, lastBody, x, y, z, h, makeConst(input), tmpOrOutput, std::forward<Interaction>(interaction),
            std::forward<Postamble>(postamble), neighborData.get(), groupDataIndex.get());
        checkGpuErrors(cudaGetLastError());

        if constexpr (Config::symmetric)
        {
            // the postamble has to be applied in a separate step for symmetric neighborhoods
            applyPostamble<Config>(firstBody, lastBody, 0, x, y, z, h, makeConst(input), makeConst(tmpOrOutput), output,
                                   std::forward<Postamble>(postamble));

            // device sync required due to possible use of allocated temporaries
            checkGpuErrors(cudaDeviceSynchronize());
        }
    }

    Statistics stats() const { return {.numBodies = lastBody - firstBody, .numBytes = numBytesUsed}; }
};

} // namespace gpu_compressed_nb_list_neighborhood_detail

template<class Config = gpu_compressed_nb_list_neighborhood_detail::GpuCompressedNbListNeighborhoodConfig<>>
struct GpuCompressedNbListNeighborhoodBuilder
{
    template<class NewCompression>
    using withCompression =
        GpuCompressedNbListNeighborhoodBuilder<typename Config::template withCompression<NewCompression>>;
    using withoutCompression = GpuCompressedNbListNeighborhoodBuilder<typename Config::withoutCompression>;
    template<bool NewSymmetric>
    using setSymmetry     = GpuCompressedNbListNeighborhoodBuilder<typename Config::template setSymmetry<NewSymmetric>>;
    using withSymmetry    = setSymmetry<true>;
    using withoutSymmetry = setSymmetry<false>;

    using Compression               = Config::Compression;
    static constexpr bool symmetric = Config::symmetric;

    unsigned ngmax;

    template<class Tc, class KeyType, class ThP>
    gpu_compressed_nb_list_neighborhood_detail::GpuCompressedNbListNeighborhood<Config, Tc, ThP>
    build(OctreeNsView<Tc, KeyType> tree,
          const Box<Tc>& box,
          const LocalIndex /*totalBodies*/,
          const GroupView& groups,
          const Tc* x,
          const Tc* y,
          const Tc* z,
          const ThP h) const
    {
        using namespace gpu_compressed_nb_list_neighborhood_detail;

        if (groups.firstBody >= groups.lastBody) return {};

        using Th = std::remove_cvref_t<std::remove_pointer_t<ThP>>;

        util::UniqueDevicePtr<Th[]> nodeRMax;
        if constexpr (Config::symmetric && std::is_pointer_v<ThP>) { nodeRMax = computeNodeRMax<Config>(tree, h); }

        const unsigned numGroups = iceil(groups.lastBody - groups.firstBody, GpuConfig::warpSize);

        // per-warp globalPool as temporary storage: first half stores uncompressed neighbor indices, second half
        // compressed indices
        constexpr unsigned warpsPerBlock = 2;
        const unsigned numBlocks         = GpuConfig::smCount * 32;
        const std::size_t totalWarps     = numBlocks * warpsPerBlock;
        const std::size_t poolSize       = totalWarps * 2 * GpuConfig::warpSize * ngmax;
        auto globalPool                  = util::deviceAlloc<std::uint32_t[]>(poolSize);

        const std::size_t maxNeighborsPerGroup    = GpuConfig::warpSize * ngmax;
        const std::size_t neighborDataVirtualSize = maxNeighborsPerGroup * numGroups;
        auto neighborData                         = util::deviceAllocVirtual<std::uint32_t[]>(neighborDataVirtualSize);
        auto groupDataIndex                       = util::deviceAlloc<std::size_t[]>(numGroups);
        auto globalBuildData                      = util::deviceAlloc<GlobalBuildData>();
        checkGpuErrors(cudaMemsetAsync(globalBuildData.get(), 0, sizeof(GlobalBuildData)));

        constexpr dim3 blockSize = {GpuConfig::warpSize, warpsPerBlock, 1};
        gpuCompressedNbListNeighborhoodBuild<Config, warpsPerBlock, Tc, ThP, KeyType><<<numBlocks, blockSize>>>(
            tree, box, groups.firstBody, groups.lastBody, numGroups, ngmax, x, y, z, h, nodeRMax.get(),
            globalPool.get(), neighborData.get(), groupDataIndex.get(), globalBuildData.get(), neighborDataVirtualSize);
        checkGpuErrors(cudaGetLastError());

        GlobalBuildData buildData;
        checkGpuErrors(cudaMemcpy(&buildData, globalBuildData.get(), sizeof(GlobalBuildData), cudaMemcpyDeviceToHost));
        switch (buildData.status)
        {
            case BuildStatus::success: break;
            case BuildStatus::neighbor_list_overflow:
                std::cerr << "WARNING: overflow in compressed neighbor list. Missing neighbors! Try to increase ngmax. "
                             "Current ngmax is "
                          << ngmax << ", but found up to " << buildData.maxNeighbors << " neighbors." << std::endl;
                break;
            case BuildStatus::neighbor_data_overflow:
                throw std::runtime_error("overflow in neighbor data in compressed nb-list neighborhood");
        }
        assert(buildData.neighborDataSize < neighborDataVirtualSize);

        return {box,
                groups.firstBody,
                groups.lastBody,
                x,
                y,
                z,
                h,
                std::move(neighborData),
                std::move(groupDataIndex),
                buildData.neighborDataSize * sizeof(std::uint32_t) + numGroups * sizeof(std::size_t)};
    }
};

} // namespace cstone::ijloop
