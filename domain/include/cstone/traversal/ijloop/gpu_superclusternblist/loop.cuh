/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Data structures and functions used for the ij-loop implementation of the supercluster neighborhood
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include "cstone/cuda/memory.cuh"
#include "cstone/compressneighbors.cuh"
#include "cstone/reducearray.cuh"
#include "cstone/traversal/ijloop/atomic_update_ptr.cuh"
#include "cstone/traversal/ijloop/common.hpp"
#include "cstone/traversal/ijloop/gpu_superclusternblist/common.cuh"
#include "cstone/util/tuple_util.hpp"

namespace cstone::ijloop::gpu_supercluster_nb_list_neighborhood_detail
{

/*! retrieve the element at the specified dynamic index from a tuple
 *
 * @param[in] tuple the tuple from which to retrieve the element
 * @param[in] index the zero-based index of the element to retrieve

 * @return the element at the specified index, cast to type of first element
 *
 * @note The caller is responsible for ensuring that the type T0 matches the type
 *       of the element at the specified index. No bounds checking is performed.
 */
template<class T0, class... T>
__device__ __forceinline__ constexpr T0 dynamicTupleGet(std::tuple<T0, T...> const& tuple, int index)
{
    T0 res;
    int i = 0;
    util::for_each_tuple(
        [&](auto const& src)
        {
            if (i++ == index) res = src;
        },
        tuple);
    return res;
}

/*! reduce and store cluster-cluster interaction results along the j-direction, i.e., computes the reduction for each
 * i-particle
 *
 * @param[in]    tuple     tuple of values to be reduced and stored
 * @param[inout] ptrs      tuple of pointers to storage locations for each value
 * @param[in]    index     index at which to store the result in the output arrays
 * @param[in]    store     boolean flag indicating whether to perform the store operation
 * @param[in]    postamble functor to apply to the data before storage
 * @param[in]    iData     particle data to be passed to the postamble
 */
template<class Config, class T0, class... T, class... Ps, class Postamble, class ParticleData>
__device__ __forceinline__ void storeTupleISum(std::tuple<T0, T...> tuple,
                                               std::tuple<Ps*...> const& ptrs,
                                               const unsigned index,
                                               const bool store,
                                               Postamble const& postamble,
                                               ParticleData const& iData)
{
    assert(blockDim.x == Config::iThreads);

    if constexpr (std::conjunction_v<std::is_same<T0, T>...> && sizeof...(T) < GpuConfig::warpSize / Config::iThreads &&
                  std::is_same<Postamble, detail::EmptyPostamble>())
    {
        // fast path: specialized reduction on multiple elements at the same time if all types in the tuple are the same

        const T0 res =
            reduceTuple<GpuConfig::warpSize / Config::iThreads, true>(tuple,
                                                                      [](auto result, auto const& value)
                                                                      {
                                                                          detail::updateResultImpl(result, value);
                                                                          return result;
                                                                      });
        if ((threadIdx.y % (GpuConfig::warpSize / Config::iThreads) <= sizeof...(T)) & store)
        {
            auto* ptr = dynamicTupleGet(ptrs, threadIdx.y % (GpuConfig::warpSize / Config::iThreads));
            if constexpr (Config::symmetric | (Config::numWarpsPerInteraction > 1))
                atomicUpdatePtr(&ptr[index], res);
            else
                ptr[index] = detail::unwrapModifiersImpl(res);
        }
    }
    else
    {
        // "slow" path: standard shuffle-based reduction for each tuple element

#pragma unroll
        for (unsigned offset = GpuConfig::warpSize / 2; offset >= Config::iThreads; offset /= 2)
            util::for_each_tuple([&](auto& t) { detail::updateResultImpl(t, shflDownSync(t, offset)); }, tuple);

        if ((threadIdx.y % (GpuConfig::warpSize / Config::iThreads) == 0) & store)
        {
            if constexpr (Config::symmetric | Config::numWarpsPerInteraction > 1)
            {
                util::for_each_tuple([index](auto* ptr, auto const& t) { atomicUpdatePtr(&ptr[index], t); }, ptrs,
                                     tuple);
            }
            else { storeParticleData(ptrs, index, postamble(iData, unwrapModifiers(tuple))); }
        }
    }
}

/*! reduce and store cluster-cluster interaction results along the i-direction, i.e., computes the reduction for each
 * j-particle
 *
 * @param[in]    tuple     tuple of values to be reduced and stored
 * @param[inout] ptrs      tuple of pointers to storage locations for each value
 * @param[in]    index     index at which to store the result in the output arrays
 * @param[in]    store     boolean flag indicating whether to perform the store operation
 */
template<class Config, class T0, class... T, class... Ps>
constexpr __device__ __forceinline__ void
storeTupleJSum(std::tuple<T0, T...> tuple, std::tuple<Ps*...> const& ptrs, const unsigned index, const bool store)
{
    assert(blockDim.x == Config::iThreads);

    if constexpr (std::conjunction_v<std::is_same<T0, T>...> && sizeof...(T) < Config::iThreads)
    {
        // fast path: specialized reduction on multiple elements at the same time if all types in the tuple are the same

        const T0 res = reduceTuple<Config::iThreads, false>(tuple,
                                                            [](auto result, auto const& value)
                                                            {
                                                                detail::updateResultImpl(result, value);
                                                                return result;
                                                            });
        if ((threadIdx.x <= sizeof...(T)) & store)
        {
            auto* ptr = dynamicTupleGet(ptrs, threadIdx.x);
            atomicUpdatePtr(&ptr[index], res);
        }
    }
    else
    {
        // "slow" path: standard shuffle-based reduction for each tuple element

#pragma unroll
        for (unsigned offset = Config::iThreads / 2; offset >= 1; offset /= 2)
            util::for_each_tuple([&](auto& t) { detail::updateResultImpl(t, shflDownSync(t, offset)); }, tuple);

        if ((threadIdx.x == 0) & store)
            util::for_each_tuple([index](auto* ptr, auto const& t) { atomicUpdatePtr(&ptr[index], t); }, ptrs, tuple);
    }
}

/*! compile-time utility to get an array buffer type for each tuple element */
template<std::size_t Size, class... Ts>
consteval std::tuple<std::array<Ts, Size>...> buffersForResults(std::tuple<Ts...> const&)
{
    return {};
}

/*! compute the amount of shared memory required per supercluster for the ij-loop kernel
 *
 * @param[in] ncmax maximum number of neighbor clusters for any supercluster
 * @return          total shared memory required per supercluster, in bytes
 */
template<class Config, class Tc, class Th, class In, class Interaction>
constexpr unsigned runIjLoopSharedMemPerSupercluster(unsigned ncmax)
{
    using ParticleData = decltype(loadParticleData((Tc*)0, (Tc*)0, (Tc*)0, (Th*)0, In{}, 0));
    using Result =
        std::decay_t<decltype(std::declval<Interaction>()(ParticleData(), ParticleData(), Vec3<Tc>(), Tc(0)))>;

    constexpr unsigned iSuperclusterDataSize =
        (Config::iClustersPerSupercluster * Config::iSize) * sizeof(ParticleData);
    unsigned nbDataSize = (ncmax + masksSize<Config>(ncmax)) * sizeof(unsigned);
    constexpr unsigned outputBuffersSize =
        !Config::symmetric && Config::numWarpsPerInteraction > 1
            ? sizeof(decltype(buffersForResults<Config::superclusterSize>(unwrapModifiers(Result()))))
            : 0;

    return iSuperclusterDataSize + nbDataSize + outputBuffersSize;
}

template<class Config, class ParticleData, class Tc, class Th, class Input>
__device__ __forceinline__ util::SharedMemAllocator::SharedMemPtr<ParticleData[]>
loadSuperclusterIParticleData(util::SharedMemAllocator& sharedAllocator,
                              const LocalIndex firstValidBody,
                              const LocalIndex totalBodies,
                              const LocalIndex iSupercluster,
                              const Tc* const __restrict__ x,
                              const Tc* const __restrict__ y,
                              const Tc* const __restrict__ z,
                              const Th* const __restrict__ h,
                              Input&& input)

{
    auto iSuperclusterData = sharedAllocator.alloc<ParticleData[]>(Config::iClustersPerSupercluster * Config::iSize);
    {
        const unsigned base = iSupercluster * Config::superclusterSize;
#pragma unroll
        for (unsigned offset = threadIdx.y * Config::iThreads + threadIdx.x; offset < Config::superclusterSize;
             offset += Config::iThreads * Config::jSize)
        {
            const unsigned i = base + offset;
            auto iData       = (i >= firstValidBody & i < totalBodies)
                                   ? loadParticleData(x, y, z, h, std::forward<Input>(input), i)
                                   : dummyParticleData(x, y, z, h, std::forward<Input>(input), i);
            std::get<0>(iData) -= firstValidBody;
            iSuperclusterData[offset] = iData;
        }
    }
    return iSuperclusterData;
}

template<class Config>
__device__ __forceinline__ std::tuple<util::SharedMemAllocator::SharedMemPtr<unsigned[]>, unsigned>
loadSuperclusterNeighborData(util::SharedMemAllocator& sharedAllocator,
                             const unsigned warpIndex,
                             const LocalIndex iSuperclusterDataIndex,
                             const unsigned iSuperclusterNeighborsCount,
                             const std::uint32_t* const __restrict__ neighborData,
                             const unsigned ncmax)
{
    auto nbData             = sharedAllocator.alloc<unsigned[]>(ncmax + masksSize<Config>(ncmax));
    const unsigned maskSize = masksSize<Config>(iSuperclusterNeighborsCount);

    if constexpr (Config::compress)
    {
        for (unsigned n = threadIdx.y * Config::iThreads + threadIdx.x; n < maskSize;
             n += Config::iThreads * Config::jSize)
            nbData[n] = neighborData[iSuperclusterDataIndex + n];
        if (warpIndex == 0)
        {
            unsigned n;
            warpDecompressNeighbors((const char*)&neighborData[iSuperclusterDataIndex + maskSize], &nbData[maskSize],
                                    n);
            assert(n == iSuperclusterNeighborsCount);
        }
    }
    else
    {
        const unsigned nbDataSize = iSuperclusterNeighborsCount + maskSize;
        for (unsigned n = threadIdx.y * Config::iThreads + threadIdx.x; n < nbDataSize;
             n += Config::iThreads * Config::jSize)
            nbData[n] = neighborData[iSuperclusterDataIndex + n];
    }

    return {std::move(nbData), maskSize};
}

template<class Config,
         unsigned NumSuperclustersPerBlock,
         bool UsePbc,
         class Tc,
         class Th,
         class In,
         class Out,
         class Interaction,
         class Postamble,
         class Mask = void>
__global__ __launch_bounds__(Config::iThreads* Config::jSize* NumSuperclustersPerBlock) void runIjLoopKernel(
    const Box<Tc> __grid_constant__ box,
    const LocalIndex firstValidBody,
    const LocalIndex totalBodies,
    const LocalIndex firstBody,
    const LocalIndex lastBody,
    const Tc* const __restrict__ x,
    const Tc* const __restrict__ y,
    const Tc* const __restrict__ z,
    const Th* const __restrict__ h,
    const In __grid_constant__ input,
    const Out __grid_constant__ output,
    const Interaction interaction,
    const Postamble postamble,
    const std::uint32_t* const __restrict__ neighborData,
    const SuperclusterInfo* const __restrict__ superclusterInfo,
    const Mask* const __restrict__ activeMasks,
    const unsigned ncmax)
{
    assert(ncmax % GpuConfig::warpSize == 0);
    static_assert(NumSuperclustersPerBlock > 0);
    static_assert(Config::iThreads * Config::jSize >= GpuConfig::warpSize);
    static_assert(Config::iThreads * Config::jSize % GpuConfig::warpSize == 0);

    assert(blockDim.x == Config::iThreads);
    assert(blockDim.y == Config::jSize);
    assert(blockDim.z == NumSuperclustersPerBlock);

    constexpr unsigned iClustersPerWarp = Config::iThreads / Config::iSize;

    const unsigned warpIndex =
        Config::numWarpsPerInteraction == 1 ? 0 : threadIdx.y / (Config::jSize / Config::numWarpsPerInteraction);

    const unsigned firstISupercluster = superclusterIndex<Config>(firstBody);
    const unsigned lastISupercluster  = superclusterIndex<Config>(lastBody - 1) + 1;
    const unsigned numISuperclusters  = lastISupercluster - firstISupercluster;
    const unsigned iSuperclusterIndex = blockIdx.x * NumSuperclustersPerBlock + threadIdx.z;
    if (iSuperclusterIndex >= numISuperclusters) return;

    const auto [iSupercluster, iSuperclusterNeighborsCount, iSuperclusterDataIndex] =
        superclusterInfo[iSuperclusterIndex];

    using ParticleData = decltype(loadParticleData(x, y, z, h, input, firstBody));
    using Result       = std::decay_t<decltype(interaction(ParticleData(), ParticleData(), Vec3<Tc>(), Tc(0)))>;

    util::SharedMemAllocator sharedAllocator(runIjLoopSharedMemPerSupercluster<Config, Tc, Th, In, Interaction>(ncmax),
                                             threadIdx.z);

    const auto iSuperclusterData = loadSuperclusterIParticleData<Config, ParticleData>(
        sharedAllocator, firstValidBody, totalBodies, iSupercluster, x, y, z, h, input);

    const auto [nbData, maskSize] = loadSuperclusterNeighborData<Config>(
        sharedAllocator, warpIndex, iSuperclusterDataIndex, iSuperclusterNeighborsCount, neighborData, ncmax);

    __syncthreads();

    std::array<Result, Config::iClustersPerSupercluster / iClustersPerWarp> iResults = {};
    const unsigned iClusterOffset = iClustersPerWarp == 1 ? 0 : threadIdx.x / Config::iSize;

    for (unsigned nb = 0; nb < iSuperclusterNeighborsCount; ++nb)
    {
        const unsigned maskStartIndex = nb * (Config::iClustersPerSupercluster * Config::numWarpsPerInteraction) +
                                        (warpIndex * Config::iClustersPerSupercluster);
        unsigned warpMask =
            (nbData[maskStartIndex / 32] >> (maskStartIndex % 32)) & ((1 << Config::iClustersPerSupercluster) - 1);

        if (warpMask)
        {
            const unsigned jCluster      = nb < iSuperclusterNeighborsCount ? nbData[nb + maskSize] : ~0u;
            const unsigned j             = jCluster * Config::jSize + threadIdx.y;
            const unsigned jSupercluster = superclusterIndex<Config>(j);
            auto jData                   = (nb < iSuperclusterNeighborsCount & j >= firstValidBody & j < totalBodies)
                                               ? loadParticleData(x, y, z, h, input, j)
                                               : dummyParticleData(x, y, z, h, input, j);
            const auto jRadiusSq         = radiusSq(jData);
            std::get<0>(jData) -= firstValidBody;
            Result jResult = {};

            warpMask >>= iClusterOffset;
#pragma unroll
            for (unsigned c = 0; c < Config::iClustersPerSupercluster; c += iClustersPerWarp)
            {
                const unsigned i = iSupercluster * Config::superclusterSize + c * Config::iSize + threadIdx.x;
                if ((warpMask >> c) & (!Config::symmetric | (iSupercluster != jSupercluster) | (i <= j)))
                {
                    bool jRequired   = i != j;
                    const auto iData = iSuperclusterData[c * Config::iSize + threadIdx.x];
                    if (!jRequired) jRequired = (i < firstBody) | (i >= lastBody);
                    assert(std::get<0>(iData) == i - firstValidBody);
                    const auto [ijPosDiff, distSq] = posDiffAndDistSq(UsePbc, box, iData, jData);
                    const bool iClose              = distSq < radiusSq(iData);
                    const bool jClose              = Config::symmetric && (distSq < jRadiusSq & jRequired);
                    if (iClose | jClose)
                    {
                        const auto ijInteraction = interaction(iData, jData, ijPosDiff, distSq);
                        if (iClose) updateResult(iResults[c / iClustersPerWarp], ijInteraction);
                        if (jClose)
                        {
                            const auto jiInteraction =
                                selectSymmetric(ijInteraction, interaction(jData, iData, -ijPosDiff, distSq));
                            updateResult(jResult, jiInteraction);
                        }
                    }
                }
            }

            if constexpr (Config::symmetric)
            {
                storeTupleJSum<Config>(jResult, output, j, j >= firstBody & j < lastBody);
            }
        }
    }

    auto activeMask = ~Config::SuperclusterParticleMask(0);
    if constexpr (!std::is_same_v<Mask, void>) activeMask = activeMasks[iSupercluster - firstISupercluster];

    if constexpr (!Config::symmetric && Config::numWarpsPerInteraction > 1)
    {
        auto outputBuffers =
            sharedAllocator.alloc<decltype(buffersForResults<Config::superclusterSize>(unwrapModifiers(Result())))>();
        auto outputBufferPtrs = util::tupleMap([](auto& array) { return array.data(); }, *outputBuffers);
        auto init             = unwrapModifiers(Result{});
#pragma unroll
        for (unsigned offset = threadIdx.y * Config::iThreads + threadIdx.x; offset < Config::superclusterSize;
             offset += Config::iThreads * Config::jSize)
            storeParticleData(outputBufferPtrs, offset, init);

        __syncthreads();

#pragma unroll
        for (unsigned c = 0; c < Config::iClustersPerSupercluster; c += iClustersPerWarp)
        {
            storeTupleISum<Config>(iResults[c / iClustersPerWarp], outputBufferPtrs, c * Config::iSize + threadIdx.x,
                                   true, detail::EmptyPostamble{}, iSuperclusterData[c * Config::iSize + threadIdx.x]);
        }

        __syncthreads();

        const unsigned base = iSupercluster * Config::superclusterSize;
#pragma unroll
        for (unsigned offset = threadIdx.y * Config::iThreads + threadIdx.x; offset < Config::superclusterSize;
             offset += Config::iThreads * Config::jSize)
        {
            const unsigned i  = base + offset;
            const bool active = (activeMask >> offset) & 1;
            if (i >= firstBody & i < lastBody & active)
            {
                const auto iData   = iSuperclusterData[offset];
                const auto iResult = util::tupleMap([&](auto const* ptr) { return ptr[offset]; }, outputBufferPtrs);
                storeParticleData(output, i, postamble(iData, unwrapModifiers(iResult)));
            }
        }
    }
    else
    {
#pragma unroll
        for (unsigned c = 0; c < Config::iClustersPerSupercluster; c += iClustersPerWarp)
        {
            const auto i      = iSupercluster * Config::superclusterSize + c * Config::iSize + threadIdx.x;
            const bool active = (activeMask >> (c * Config::iSize + threadIdx.x)) & 1;
            storeTupleISum<Config>(iResults[c / iClustersPerWarp], output, i, i >= firstBody & i < lastBody & active,
                                   postamble, iSuperclusterData[c * Config::iSize + threadIdx.x]);
        }
    }
}

template<class Config,
         class Tc,
         class Th,
         class Input,
         class Output,
         class Interaction,
         class Postamble,
         class Mask = void>
void runIjLoop(const Box<Tc>& box,
               const LocalIndex firstValidBody,
               const LocalIndex totalBodies,
               const LocalIndex firstBody,
               const LocalIndex lastBody,
               const Tc* const x,
               const Tc* const y,
               const Tc* const z,
               const Th* const h,
               Input&& input,
               Output&& output,
               Interaction&& interaction,
               Postamble&& postamble,
               const std::uint32_t* const neighborData,
               const SuperclusterInfo* const superclusterInfo,
               const LocalIndex numISuperclusters,
               const Mask* const activeMasks,
               const unsigned ncmax)
{
    constexpr unsigned numSuperclustersPerBlock = 64 / (Config::iThreads * Config::jSize);
    const dim3 blockSize                        = {Config::iThreads, Config::jSize, numSuperclustersPerBlock};
    const unsigned numBlocks                    = iceil(numISuperclusters, numSuperclustersPerBlock);
    const unsigned sharedMem =
        numSuperclustersPerBlock *
        runIjLoopSharedMemPerSupercluster<Config, Tc, Th, std::decay_t<decltype(makeConstRestrict(input))>,
                                          std::decay_t<Interaction>>(ncmax);
    const auto run = [&](auto usePbc)
    {
        runIjLoopKernel<Config, numSuperclustersPerBlock, decltype(usePbc)::value><<<numBlocks, blockSize, sharedMem>>>(
            box, firstValidBody, totalBodies, firstBody, lastBody, x, y, z, h, std::forward<Input>(input),
            std::forward<Output>(output), std::forward<Interaction>(interaction), std::forward<Postamble>(postamble),
            neighborData, superclusterInfo, activeMasks, ncmax);
        checkGpuErrors(cudaGetLastError());
    };

    if (box.boundaryX() == BoundaryType::periodic | box.boundaryY() == BoundaryType::periodic |
        box.boundaryZ() == BoundaryType::periodic)
        run(std::true_type());
    else
        run(std::false_type());
}

template<class Tc, class Th, class In, class Out, class Interaction>
__global__ void initResultKernel(const LocalIndex firstBody,
                                 const LocalIndex lastBody,
                                 const Tc* __restrict__ x,
                                 const Tc* __restrict__ y,
                                 const Tc* __restrict__ z,
                                 const Th* __restrict__ h,
                                 const In __grid_constant__ input,
                                 const Out __grid_constant__ output,
                                 Interaction interaction)
{
    const LocalIndex i = blockDim.x * blockIdx.x + threadIdx.x + firstBody;
    if (i >= lastBody) return;

    using ParticleData = decltype(loadParticleData(x, y, z, h, input, firstBody));
    using Result       = decltype(interaction(ParticleData{}, ParticleData{}, Vec3<Tc>{0, 0, 0}, Tc(0)));
    storeParticleData(output, i, unwrapModifiers(Result{}));
}

template<class Config, class Tc, class Th, class Input, class Output, class Interaction>
void initResult(const LocalIndex firstBody,
                const LocalIndex lastBody,
                const Tc* x,
                const Tc* y,
                const Tc* z,
                const Th* h,
                Input&& input,
                Output&& output,
                Interaction&& interaction)
{
    static_assert(Config::symmetric);
    const LocalIndex numBodies = lastBody - firstBody;
    constexpr unsigned threads = 256;
    const unsigned numBlocks   = iceil(numBodies, threads);
    initResultKernel<<<numBlocks, threads>>>(firstBody, lastBody, x, y, z, h, std::forward<Input>(input),
                                             std::forward<Output>(output), std::forward<Interaction>(interaction));
    checkGpuErrors(cudaGetLastError());
}

template<class Tc, class Th, class In, class Tmp, class Out, class Postamble>
__global__ void applyPostambleKernel(const LocalIndex firstBody,
                                     const LocalIndex lastBody,
                                     const LocalIndex firstValidBody,
                                     const Tc* __restrict__ x,
                                     const Tc* __restrict__ y,
                                     const Tc* __restrict__ z,
                                     const Th* __restrict__ h,
                                     const In __grid_constant__ input,
                                     const Tmp __grid_constant__ tmp,
                                     const Out __grid_constant__ output,
                                     const Postamble postamble)
{
    const LocalIndex i = blockDim.x * blockIdx.x + threadIdx.x + firstBody;
    if (i >= lastBody) return;

    auto iData = loadParticleData(x, y, z, h, input, i);
    std::get<0>(iData) -= firstValidBody;
    const auto result = util::tupleMap([&](auto* ptr) { return ptr[i]; }, tmp);
    storeParticleData(output, i, postamble(iData, result));
}

template<class Config, class Tc, class Th, class Input, class Tmp, class Output, class Postamble>
void applyPostamble(const LocalIndex firstBody,
                    const LocalIndex lastBody,
                    const LocalIndex firstValidBody,
                    const Tc* x,
                    const Tc* y,
                    const Tc* z,
                    const Th* h,
                    Input&& input,
                    Tmp&& tmp,
                    Output&& output,
                    Postamble&& postamble)
{
    static_assert(Config::symmetric);

    if constexpr (std::is_same_v<std::remove_cvref_t<Postamble>, detail::EmptyPostamble>) return;

    const LocalIndex numBodies = lastBody - firstBody;
    constexpr unsigned threads = 256;
    const unsigned numBlocks   = iceil(numBodies, threads);
    applyPostambleKernel<<<numBlocks, threads>>>(firstBody, lastBody, firstValidBody, x, y, z, h,
                                                 std::forward<Input>(input), std::forward<Tmp>(tmp),
                                                 std::forward<Output>(output), std::forward<Postamble>(postamble));
    checkGpuErrors(cudaGetLastError());
}

template<class Config>
__global__ void computeActiveMasksKernel(const LocalIndex firstISupercluster,
                                         const LocalIndex firstValidBody,
                                         const GroupView __grid_constant__ groups,
                                         typename Config::SuperclusterParticleMask* __restrict__ activeMasks)
{
    using Mask = typename Config::SuperclusterParticleMask;

    const LocalIndex index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= groups.numGroups) return;

    const LocalIndex groupStart = groups.groupStart[index] + firstValidBody;
    const LocalIndex groupEnd   = groups.groupEnd[index] + firstValidBody;
    assert(groupStart < groupEnd);
    assert(superclusterIndex<Config>(groupStart) == superclusterIndex<Config>(groupEnd - 1));

    const LocalIndex supercluster = superclusterIndex<Config>(groupStart);
    const LocalIndex startOffset  = groupStart - supercluster * Config::superclusterSize;
    const LocalIndex endOffset    = groupEnd - supercluster * Config::superclusterSize;
    assert(startOffset < GpuConfig::warpSize);
    assert(endOffset <= GpuConfig::warpSize);

    auto* activeMaskPtr   = &activeMasks[supercluster - firstISupercluster];
    const Mask activeMask = ~(~Mask(0) << endOffset) & (~Mask(0) << startOffset);

    // atomic update as multiple groups can be inside the same supercluster
    atomicOr(activeMaskPtr, activeMask);
}

template<class Config>
util::UniqueDevicePtr<typename Config::SuperclusterParticleMask[]>
computeActiveMasks(const LocalIndex firstISupercluster,
                   const LocalIndex numISuperclusters,
                   const LocalIndex firstValidBody,
                   const GroupView& groups)
{
    auto activeMasks = util::deviceAlloc<typename Config::SuperclusterParticleMask[]>(numISuperclusters);
    checkGpuErrors(
        cudaMemsetAsync(activeMasks.get(), 0, sizeof(typename Config::SuperclusterParticleMask) * numISuperclusters));

    constexpr unsigned numThreads = 256;
    const unsigned numBlocks      = iceil(groups.numGroups, numThreads);
    computeActiveMasksKernel<Config>
        <<<numBlocks, numThreads>>>(firstISupercluster, firstValidBody, groups, activeMasks.get());
    checkGpuErrors(cudaGetLastError());
    return activeMasks;
}

} // namespace cstone::ijloop::gpu_supercluster_nb_list_neighborhood_detail
