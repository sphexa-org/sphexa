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
 * @brief GROMACS-like neighbor search
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#include <array>
#include <map>
#include <mutex>
#include <shared_mutex>

#include <cooperative_groups.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/universal_vector.h>

#include "cstone/cuda/gpu_config.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/reducearray.cuh"
#include "cstone/traversal/ijloop/ijloop.hpp"
#include "cstone/tree/octree.hpp"

namespace cstone::ijloop
{

namespace gromacs_like_neighborhood_detail
{

// Constants from GROMACS

constexpr unsigned clusterSize               = 8;
constexpr unsigned clusterPairSplit          = GpuConfig::warpSize == 64 ? 1 : 2;
constexpr unsigned numClusterPerSupercluster = 8;
constexpr unsigned jGroupSize                = 32 / numClusterPerSupercluster;
constexpr unsigned superClusterSize          = numClusterPerSupercluster * clusterSize;
constexpr unsigned exclSize                  = clusterSize * clusterSize / clusterPairSplit;

// Gromacs-like data structures

struct Sci
{
    unsigned sci, cjPackedBegin, cjPackedEnd;

    constexpr bool operator<(Sci const& other) const
    {
        return (cjPackedEnd - cjPackedBegin) > (other.cjPackedEnd - other.cjPackedBegin);
    }
};

struct Excl
{
    std::array<unsigned, exclSize> pair = {0};

    bool operator==(Excl const& other) const { return pair == other.pair; }
    bool operator<(Excl const& other) const { return pair < other.pair; }
};

struct ImEi
{
    unsigned imask   = 0u;
    unsigned exclInd = 0u;
};

struct CjPacked
{
    std::array<unsigned, jGroupSize> cj;
    std::array<ImEi, clusterPairSplit> imei;
};

// Helpers for building the GROMACS-like NB lists

constexpr bool includeNb(unsigned i, unsigned j)
{
    const unsigned ci    = i / clusterSize;
    const unsigned sci   = ci / numClusterPerSupercluster;
    const unsigned cj    = j / clusterSize;
    const unsigned scj   = cj / numClusterPerSupercluster;
    const bool s         = sci % 2 == scj % 2;
    const bool includeSc = sci < scj ? s : !s;
    return includeSc || (sci == scj && (ci < cj || (ci == cj && i <= j)));
}

struct CjData
{
    Excl excl;
    unsigned imask;
};

template<class Tc, class Th, class KeyType>
std::map<unsigned, std::array<CjData, clusterPairSplit>>
clusterNeighborsOfSuperCluster(OctreeNsView<Tc, KeyType> const& tree,
                               Box<Tc> const& box,
                               const Tc* x,
                               const Tc* y,
                               const Tc* z,
                               const Th* h,
                               const unsigned lastBody,
                               const unsigned ngmax,
                               LocalIndex* neighbors,
                               const unsigned sci)
{
    constexpr unsigned splitClusterSize = clusterSize / clusterPairSplit;
    std::map<unsigned, std::array<CjData, clusterPairSplit>> superClusterNeighbors;
    for (unsigned cii = 0; cii < numClusterPerSupercluster; ++cii)
    {
        const unsigned ci = sci * numClusterPerSupercluster + cii;
        for (unsigned ii = 0; ii < clusterSize; ++ii)
        {
            const unsigned i = ci * clusterSize + ii;
            if (i >= lastBody) break;
            unsigned nci = std::min(findNeighbors(i, x, y, z, h, tree, box, ngmax, neighbors), ngmax);
            if (nci < ngmax) neighbors[nci++] = i;
            for (unsigned nb = 0; nb < nci; ++nb)
            {
                const unsigned j = neighbors[nb];
                if (includeNb(i, j))
                {
                    const unsigned cj    = j / clusterSize;
                    const unsigned jj    = j - cj * clusterSize;
                    auto [it, _]         = superClusterNeighbors.emplace(cj, std::array<CjData, clusterPairSplit>({}));
                    const unsigned split = jj / splitClusterSize;
                    it->second.at(split).imask |= 1 << cii;
                    const unsigned thread = ii + (jj % splitClusterSize) * clusterSize;
                    it->second.at(split).excl.pair[thread] |= 1 << cii;
                }
            }
        }
    }
    return superClusterNeighbors;
}

void optimizeExcl(const unsigned lastBody,
                  const unsigned sci,
                  const CjPacked& cjPacked,
                  std::array<Excl, clusterPairSplit>& excl)
{
    // Keep exact interaction on last super cluster to avoid OOB accesses
    const unsigned numSuperclusters = iceil(lastBody, superClusterSize);
    if (sci == numSuperclusters - 1) return;

    const unsigned numClusters = iceil(lastBody, clusterSize);
    bool selfInteraction       = false;
    for (unsigned cj : cjPacked.cj)
    {
        // Keep exact interaction on last j-cluster to avoid OOB accesses
        if (cj == numClusters - 1) return;
        if (cj / numClusterPerSupercluster == sci) selfInteraction = true;
    }

    if (!selfInteraction)
    {
        // Compute all interactions if j-clusters do not overlap super cluster
        for (unsigned split = 0; split < clusterPairSplit; ++split)
            excl[split].pair.fill(0xffffffffu);
        return;
    }

    // Use triangular masks (i < j) where j-clusters overlap with super cluster
    for (unsigned jj = 0; jj < clusterSize; ++jj)
    {
        for (unsigned ii = 0; ii < clusterSize; ++ii)
        {
            const unsigned thread = ii + (jj % (clusterSize / clusterPairSplit)) * clusterSize;
            const unsigned split  = jj / (clusterSize / clusterPairSplit);
            for (unsigned jm = 0; jm < jGroupSize; ++jm)
            {
                for (unsigned cii = 0; cii < numClusterPerSupercluster; ++cii)
                {
                    const unsigned ci = sci * numClusterPerSupercluster + cii;
                    const unsigned i  = ci * clusterSize + ii;
                    const unsigned j  = cjPacked.cj[jm] * clusterSize + jj;
                    excl[split].pair[thread] |= includeNb(i, j) ? 1 << (cii + jm * numClusterPerSupercluster) : 0;
                }
            }
        }
    }
}

template<class T>
constexpr __forceinline__ void atomicAddScalarOrVec(T* ptr, T value)
{
    atomicAdd(ptr, value);
}

template<class T, std::size_t N>
constexpr __forceinline__ void atomicAddScalarOrVec(util::array<T, N>* ptr, util::array<T, N> const& value)
{
#pragma unroll
    for (std::size_t i = 0; i < N; ++i)
        atomicAddScalarOrVec(&((*ptr)[i]), value[i]);
}

template<class T0, class... T>
__device__ inline constexpr T0 dynamicTupleGet(std::tuple<T0, T...> const& tuple, int index)
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

template<class T0, class... T>
__device__ __forceinline__ void
storeTupleISum(std::tuple<T0, T...> tuple, std::tuple<T0*, T*...> const& ptrs, const unsigned index, const bool store)
{
    const auto block = cooperative_groups::this_thread_block();
    assert(block.dim_threads().x == clusterSize);
    const auto warp                     = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);
    constexpr unsigned splitClusterSize = clusterSize / clusterPairSplit;

    if constexpr (std::conjunction_v<std::is_same<T0, T>...> && sizeof...(T) < splitClusterSize)
    {
        const T0 res = reduceTuple<splitClusterSize, true>(tuple, std::plus<T0>());
        if ((block.thread_index().y % splitClusterSize <= sizeof...(T)) & store)
        {
            T0* ptr = dynamicTupleGet(ptrs, block.thread_index().y % splitClusterSize);
            atomicAdd(&ptr[index], res);
        }
    }
    else
    {
#pragma unroll
        for (unsigned offset = GpuConfig::warpSize / 2; offset >= clusterSize; offset /= 2)
            util::for_each_tuple([&](auto& t) { t += warp.shfl_down(t, offset); }, tuple);

        if ((block.thread_index().y % splitClusterSize == 0) & store)
            util::for_each_tuple([index](auto* ptr, auto const& t) { atomicAddScalarOrVec(&ptr[index], t); }, ptrs,
                                 tuple);
    }
}

template<class T0, class... T>
constexpr __device__ void
storeTupleJSum(std::tuple<T0, T...> tuple, std::tuple<T0*, T*...> const& ptrs, const unsigned index, bool store)
{
    const auto block = cooperative_groups::this_thread_block();
    assert(block.dim_threads().x == clusterSize);
    const auto warp = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);

    if constexpr (std::conjunction_v<std::is_same<T0, T>...> && sizeof...(T) < clusterSize)
    {
        const T0 res = reduceTuple<clusterSize, false>(tuple, std::plus<T0>());
        if ((block.thread_index().x <= sizeof...(T)) & store)
        {
            T0* ptr = dynamicTupleGet(ptrs, block.thread_index().x);
            atomicAddScalarOrVec(&ptr[index], res);
        }
    }
    else
    {
#pragma unroll
        for (unsigned offset = clusterSize / 2; offset >= 1; offset /= 2)
            util::for_each_tuple([&](auto& t) { t += warp.shfl_down(t, offset); }, tuple);

        if ((block.thread_index().x == 0) & store)
            util::for_each_tuple([index](auto* ptr, auto const& t) { atomicAddScalarOrVec(&ptr[index], t); }, ptrs,
                                 tuple);
    }
}

template<bool UsePbc, Symmetry Sym, class Tc, class Th, class In, class Out, class Interaction>
__global__
__launch_bounds__(clusterSize* clusterSize) void gromacsLikeNeighborhoodKernel(const Box<Tc> __grid_constant__ box,
                                                                               const LocalIndex firstBody,
                                                                               const LocalIndex lastBody,
                                                                               const Tc* __restrict__ x,
                                                                               const Tc* __restrict__ y,
                                                                               const Tc* __restrict__ z,
                                                                               const Th* __restrict__ h,
                                                                               const In __grid_constant__ input,
                                                                               const Out __grid_constant__ output,
                                                                               const Interaction interaction,
                                                                               const Sci* __restrict__ sciSorted,
                                                                               const CjPacked* __restrict__ cjPacked,
                                                                               const Excl* __restrict__ excl)
{
    constexpr unsigned superClusterInteractionMask = (1u << numClusterPerSupercluster) - 1u;

    const auto block = cooperative_groups::this_thread_block();
    const auto warp  = cooperative_groups::tiled_partition<GpuConfig::warpSize>(block);

    const Sci nbSci               = sciSorted[block.group_index().x];
    const unsigned sci            = nbSci.sci;
    const unsigned cijPackedBegin = nbSci.cjPackedBegin;
    const unsigned cijPackedEnd   = nbSci.cjPackedEnd;

    using particleData_t = decltype(loadParticleData(x, y, z, h, input, 0));

    __shared__ particleData_t xqib[clusterSize * numClusterPerSupercluster];

    constexpr bool loadUsingAllXYThreads = clusterSize == numClusterPerSupercluster;
    if (loadUsingAllXYThreads || block.thread_index().y < numClusterPerSupercluster)
    {
        const unsigned ci = sci * numClusterPerSupercluster + block.thread_index().y;
        const unsigned ai = ci * clusterSize + block.thread_index().x;
        xqib[block.thread_index().y * clusterSize + block.thread_index().x] =
            loadParticleData(x, y, z, h, input, std::min(ai, lastBody - 1));
    }
    block.sync();

    using result_t = decltype(interaction(particleData_t{}, particleData_t{}, Vec3<Tc>(), Tc()));

    std::array<result_t, numClusterPerSupercluster> iResultBuf = {};

    for (unsigned jPacked = cijPackedBegin; jPacked < cijPackedEnd; ++jPacked)
    {
        const unsigned wexclIdx = cjPacked[jPacked].imei[warp.meta_group_rank()].exclInd;
        const unsigned imask    = cjPacked[jPacked].imei[warp.meta_group_rank()].imask;
        const unsigned wexcl    = excl[wexclIdx].pair[warp.thread_rank()];
        if (imask)
        {
            for (unsigned jm = 0; jm < jGroupSize; ++jm)
            {
                if (imask & (superClusterInteractionMask << (jm * numClusterPerSupercluster)))
                {
                    unsigned maskJi     = 1u << (jm * numClusterPerSupercluster);
                    const unsigned cj   = cjPacked[jPacked].cj[jm];
                    const unsigned aj   = cj * clusterSize + block.thread_index().y;
                    const auto jData    = loadParticleData(x, y, z, h, input, std::min(aj, lastBody - 1));
                    result_t jResultBuf = {};

#pragma unroll
                    for (unsigned i = 0; i < numClusterPerSupercluster; ++i)
                    {
                        if (imask & maskJi)
                        {
                            const unsigned ci = sci * numClusterPerSupercluster + i;
                            const unsigned ai = ci * clusterSize + block.thread_index().x;
                            if (ai < lastBody)
                            {
                                const auto iData               = xqib[i * clusterSize + block.thread_index().x];
                                const auto [ijPosDiff, distSq] = posDiffAndDistSq(UsePbc, box, iData, jData);
                                const Th iRadiusSq             = radiusSq(iData);
                                const Tc intBit                = (wexcl & maskJi) ? Tc(1) : Tc(0);
                                if ((distSq < iRadiusSq) * intBit)
                                {
                                    auto ijInteraction = interaction(iData, jData, ijPosDiff, distSq);
                                    updateResult(iResultBuf[i], ijInteraction);
                                    if constexpr (std::is_same_v<Sym, symmetry::Asymmetric>)
                                        ijInteraction = interaction(jData, iData, -ijPosDiff, distSq);
                                    if (ai != aj) updateResult(jResultBuf, ijInteraction);
                                }
                            }
                        }
                        maskJi += maskJi;
                    }

                    if constexpr (std::is_same_v<Sym, symmetry::Odd>)
                        util::for_each_tuple([](auto& v) { v = -v; }, jResultBuf);

                    storeTupleJSum(jResultBuf, output, aj, aj >= firstBody & aj < lastBody);
                }
            }
        }
    }

    for (unsigned i = 0; i < numClusterPerSupercluster; ++i)
    {
        const unsigned ai = (sci * numClusterPerSupercluster + i) * clusterSize + block.thread_index().x;
        storeTupleISum(iResultBuf[i], output, ai, ai >= firstBody & ai < lastBody);
    }
}

template<class Tc, class Th>
struct GromacsLikeNeighborhoodImpl
{
    thrust::universal_vector<Sci> sciSorted;
    thrust::universal_vector<CjPacked> cjPacked;
    thrust::universal_vector<Excl> excl;
    Box<Tc> box;
    LocalIndex firstBody, lastBody;
    const Tc *x, *y, *z;
    const Th* h;

    template<class... In, class... Out, class Interaction, Symmetry Sym>
    void
    ijLoop(std::tuple<In*...> const& input, std::tuple<Out*...> const& output, Interaction&& interaction, Sym) const
    {
        const auto constInput = makeConstRestrict(input);

        const LocalIndex numBodies = lastBody - firstBody;
        util::for_each_tuple(
            [&](auto* ptr) { checkGpuErrors(cudaMemsetAsync(ptr + firstBody, 0, sizeof(decltype(*ptr)) * numBodies)); },
            output);

        const dim3 blockSize     = {clusterSize, clusterSize, 1};
        const unsigned numBlocks = sciSorted.size();
        if (box.boundaryX() == BoundaryType::periodic | box.boundaryY() == BoundaryType::periodic |
            box.boundaryZ() == BoundaryType::periodic)
        {
            gromacsLikeNeighborhoodKernel<true, Sym><<<numBlocks, blockSize>>>(
                box, firstBody, lastBody, x, y, z, h, constInput, output, std::forward<Interaction>(interaction),
                rawPtr(sciSorted), rawPtr(cjPacked), rawPtr(excl));
        }
        else
        {
            gromacsLikeNeighborhoodKernel<false, Sym><<<numBlocks, blockSize>>>(
                box, firstBody, lastBody, x, y, z, h, constInput, output, std::forward<Interaction>(interaction),
                rawPtr(sciSorted), rawPtr(cjPacked), rawPtr(excl));
        }
        checkGpuErrors(cudaGetLastError());
    }

    Statistics stats() const
    {
        return {.numBodies = lastBody - firstBody,
                .numBytes  = sciSorted.size() * sizeof(typename decltype(sciSorted)::value_type) +
                            cjPacked.size() * sizeof(typename decltype(cjPacked)::value_type) +
                            excl.size() * sizeof(typename decltype(excl)::value_type)};
    }
};

} // namespace gromacs_like_neighborhood_detail

struct GromacsLikeNeighborhood
{
    unsigned ngmax;

    template<class Tc, class KeyType, class Th>
    gromacs_like_neighborhood_detail::GromacsLikeNeighborhoodImpl<Tc, Th> build(const OctreeNsView<Tc, KeyType>& tree,
                                                                                const Box<Tc>& box,
                                                                                const LocalIndex /* totalBodies */,
                                                                                const GroupView& groups,
                                                                                const Tc* x,
                                                                                const Tc* y,
                                                                                const Tc* z,
                                                                                const Th* h) const
    {
        using namespace gromacs_like_neighborhood_detail;

        assert(groups.firstBody == 0 && totalBodies == groups.lastBody);
        const unsigned numSuperclusters = iceil(groups.lastBody, superClusterSize);

        GromacsLikeNeighborhoodImpl<Tc, Th> nbList{thrust::universal_vector<Sci>(numSuperclusters),
                                                   thrust::universal_vector<CjPacked>(0),
                                                   thrust::universal_vector<Excl>(1),
                                                   box,
                                                   groups.firstBody,
                                                   groups.lastBody,
                                                   x,
                                                   y,
                                                   z,
                                                   h};
        nbList.cjPacked.reserve(numSuperclusters);
        nbList.excl.front().pair.fill(0xffffffffu);

        std::map<Excl, unsigned> exclIndexMap;
        exclIndexMap[nbList.excl.front()] = 0;

        std::shared_mutex exclMutex, cjPackedMutex;

        const auto exclIndex = [&](Excl const& e)
        {
            {
                std::shared_lock lock(exclMutex);
                const auto it = exclIndexMap.find(e);
                if (it != exclIndexMap.end()) return it->second;
            }
            {
                std::unique_lock lock(exclMutex);
                const auto it = exclIndexMap.find(e);
                if (it != exclIndexMap.end()) return it->second;
                const unsigned index = nbList.excl.size();
                nbList.excl.push_back(e);
                exclIndexMap[e] = index;
                return index;
            }
        };

#pragma omp parallel
        {
            std::vector<LocalIndex> neighbors(ngmax);
#pragma omp for
            for (unsigned sci = 0; sci < numSuperclusters; ++sci)
            {
                const auto superClusterNeighbors = clusterNeighborsOfSuperCluster(
                    tree, box, x, y, z, h, groups.lastBody, ngmax, neighbors.data(), sci);

                const unsigned ncjPacked = iceil(superClusterNeighbors.size(), jGroupSize);
                unsigned cjPackedBegin, cjPackedEnd;
                {
                    std::unique_lock lock(cjPackedMutex);
                    cjPackedBegin = nbList.cjPacked.size();
                    cjPackedEnd   = cjPackedBegin + ncjPacked;
                    nbList.cjPacked.resize(cjPackedEnd);
                }
                auto it = superClusterNeighbors.begin();
                for (unsigned n = 0; n < ncjPacked; ++n)
                {
                    CjPacked next                               = {};
                    std::array<Excl, clusterPairSplit> nextExcl = {};
                    for (unsigned jm = 0; jm < jGroupSize; ++jm, ++it)
                    {
                        if (it == superClusterNeighbors.end()) break;

                        const auto& [cj, data] = *it;
                        next.cj[jm]            = cj;
                        for (unsigned split = 0; split < clusterPairSplit; ++split)
                        {
                            next.imei[split].imask |= data[split].imask << (jm * numClusterPerSupercluster);
                            for (unsigned e = 0; e < exclSize; ++e)
                                nextExcl[split].pair[e] |= data[split].excl.pair[e] << (jm * numClusterPerSupercluster);
                        }
                    }
                    optimizeExcl(groups.lastBody, sci, next, nextExcl);
                    for (unsigned split = 0; split < clusterPairSplit; ++split)
                        next.imei[split].exclInd = exclIndex(nextExcl[split]);
                    {
                        std::shared_lock lock(cjPackedMutex);
                        nbList.cjPacked[cjPackedBegin + n] = next;
                    }
                }

                nbList.sciSorted[sci] = {sci, cjPackedBegin, cjPackedEnd};
            }
        }
        thrust::stable_sort(thrust::device, nbList.sciSorted.begin(), nbList.sciSorted.end());
        return nbList;
    }
};

} // namespace cstone::ijloop
