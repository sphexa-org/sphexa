/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Common testing infrastructure for ij-loop benchmarks
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <span>
#include <tuple>
#include <type_traits>
#include <vector>

#include <thrust/universal_vector.h>

#include "cstone/cuda/cuda_runtime.hpp"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/sfc/box.hpp"
#include "cstone/traversal/ijloop/cpu_alwaystraverse.hpp"
#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/traversal/groups.hpp"
#include "cstone/tree/octree.hpp"
#include "cstone/util/tuple_util.hpp"

struct NeighborhoodBenchmarkResults
{
    std::vector<float> runTimes;
    float buildTime;
    float numBytesPerParticle;
};

template<class Tc,
         class T,
         class StrongKeyType,
         class Coords,
         cstone::ijloop::NeighborhoodBuilder NeighborhoodBuilder,
         class Interaction,
         class... InputTs,
         class... OutputTs>
NeighborhoodBenchmarkResults benchmarkNeighborhood(const Coords& coords,
                                                   const NeighborhoodBuilder& neighborhoodBuilder,
                                                   const T hVal,
                                                   const float searchExtFactor,
                                                   unsigned ngmax,
                                                   const Interaction& interaction,
                                                   const std::tuple<InputTs...>& inputValues,
                                                   const std::tuple<OutputTs...>& initialOutputValues,
                                                   bool validate = true)
{
    using namespace cstone;
    using KeyType = typename StrongKeyType::ValueType;

    const unsigned n = coords.x().size();
    printf("Number of particles: %u\n", n);

    const std::vector<T> h(n, hVal);
    const Box<Tc> box = coords.box();

    // compute average number of neighbor particles assuming a random uniform particle distribution
    const double r                  = 2 * hVal;
    const double expected_neighbors = 4.0 / 3.0 * M_PI * r * r * r * n / (box.lx() * box.ly() * box.lz());
    printf("Expected average number of neighbors for computations: %.0f\n", expected_neighbors);
    if (searchExtFactor != 1)
    {
        const double rExt = r * searchExtFactor;
        const double expected_neighbors_in_list =
            4.0 / 3.0 * M_PI * rExt * rExt * rExt * n / (box.lx() * box.ly() * box.lz());
        printf("Expected average number of neighbors in NB lists: %.0f\n", expected_neighbors_in_list);
    }

    const Tc* x         = coords.x().data();
    const Tc* y         = coords.y().data();
    const Tc* z         = coords.z().data();
    const KeyType* keys = coords.particleKeys().data();

    // build the cornerstone octree on the CPU
    constexpr unsigned bucketSize = 64;
    const auto [csTree, counts]   = computeOctree(std::span(coords.particleKeys()), bucketSize);
    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(csTree));
    updateInternalTree<KeyType>(csTree, octree.data());

    std::vector<LocalIndex> layout(nNodes(csTree) + 1, 0);
    std::inclusive_scan(counts.begin(), counts.end(), layout.begin() + 1);

    std::vector<Vec3<Tc>> centers(octree.numNodes), sizes(octree.numNodes);
    std::span<const KeyType> nodeKeys(octree.prefixes.data(), octree.numNodes);
    nodeFpCenters<KeyType>(nodeKeys, centers.data(), sizes.data(), box);

    const OctreeNsView<Tc, KeyType> nsView{.numLeafNodes    = octree.numLeafNodes,
                                           .numNodes        = octree.numNodes,
                                           .prefixes        = octree.prefixes.data(),
                                           .childOffsets    = octree.childOffsets.data(),
                                           .parents         = octree.parents.data(),
                                           .internalToLeaf  = octree.internalToLeaf.data(),
                                           .leafToInternal  = octree.leafToInternal.data(),
                                           .levelRange      = octree.levelRange.data(),
                                           .leaves          = keys,
                                           .layout          = layout.data(),
                                           .centers         = centers.data(),
                                           .sizes           = sizes.data(),
                                           .searchExtFactor = 1};
    LocalIndex zero = 0;
    const GroupView groupView{.firstBody = 0, .lastBody = n, .numGroups = 1, .groupStart = &zero, .groupEnd = &n};

    // compute reference using basic CPU neighborhood
    const auto allocVec = [n]<class Tv>(Tv initialValue) { return std::vector<Tv>(n, initialValue); };
    const std::tuple<std::vector<InputTs>...> inputs = util::tupleMap(allocVec, inputValues);
    std::tuple<std::vector<OutputTs>...> outputs     = util::tupleMap(allocVec, initialOutputValues);
    ijloop::CpuAlwaysTraverseNeighborhoodBuilder{ngmax}
        .build(nsView, box, n, groupView, x, y, z, hVal)
        .ijLoop(util::tupleMap([](auto const& v) { return v.data(); }, inputs),
                util::tupleMap([](auto& v) { return v.data(); }, outputs), interaction, ijloop::empty_postamble);

    // allocate GPU data, use thrust::universal_vector to support neighborhoods that build on the CPU
    const thrust::universal_vector<Tc> dX(coords.x().begin(), coords.x().end()),
        dY(coords.y().begin(), coords.y().end()), dZ(coords.z().begin(), coords.z().end());
    const auto allocGpuVec = [n]<class Tv>(Tv initialValue) { return thrust::universal_vector<Tv>(n, initialValue); };
    const auto dH          = allocGpuVec(hVal);
    const std::tuple<thrust::universal_vector<InputTs>...> dInputs = util::tupleMap(allocGpuVec, inputValues);
    std::tuple<thrust::universal_vector<OutputTs>...> dOutputs     = util::tupleMap(allocGpuVec, initialOutputValues);

    // compute particle memory usage
    std::size_t particleMemoryUsage = (dX.size() + dY.size() + dZ.size()) * sizeof(Tc);
    const auto addMemoryUsage       = [&]<class Tv>(thrust::universal_vector<Tv> const& v)
    { particleMemoryUsage += v.size() * sizeof(Tv); };
    util::for_each_tuple(addMemoryUsage, dInputs);
    util::for_each_tuple(addMemoryUsage, dOutputs);
    printf("Memory usage of particle data: %.2f MB\n", particleMemoryUsage / 1.0e6);

    // move tree data to the GPU
    const thrust::universal_vector<KeyType> dPrefixes             = octree.prefixes;
    const thrust::universal_vector<TreeNodeIndex> dChildOffsets   = octree.childOffsets;
    const thrust::universal_vector<TreeNodeIndex> dParents        = octree.parents;
    const thrust::universal_vector<TreeNodeIndex> dInternalToLeaf = octree.internalToLeaf;
    const thrust::universal_vector<TreeNodeIndex> dLeafToInternal = octree.leafToInternal;
    const thrust::universal_vector<TreeNodeIndex> dLevelRange     = octree.levelRange;
    const thrust::universal_vector<LocalIndex> dLayout            = layout;
    const thrust::universal_vector<Vec3<Tc>> dCenters             = centers;
    const thrust::universal_vector<Vec3<Tc>> dSizes               = sizes;
    printf("Memory usage of tree data: %.2f MB\n",
           (sizeof(KeyType) * dPrefixes.size() +
            sizeof(TreeNodeIndex) * (dChildOffsets.size() + dInternalToLeaf.size() + dLevelRange.size()) +
            sizeof(LocalIndex) * dLayout.size() + sizeof(Vec3<Tc>) * (dCenters.size() + dSizes.size())) /
               1.0e6);

    const thrust::universal_vector<KeyType> dCodes(coords.particleKeys().begin(), coords.particleKeys().end());
    const OctreeNsView<Tc, KeyType> dNsView{.numLeafNodes    = octree.numLeafNodes,
                                            .numNodes        = octree.numNodes,
                                            .prefixes        = rawPtr(dPrefixes),
                                            .childOffsets    = rawPtr(dChildOffsets),
                                            .parents         = rawPtr(dParents),
                                            .internalToLeaf  = rawPtr(dInternalToLeaf),
                                            .leafToInternal  = rawPtr(dLeafToInternal),
                                            .levelRange      = rawPtr(dLevelRange),
                                            .leaves          = rawPtr(dCodes),
                                            .layout          = rawPtr(dLayout),
                                            .centers         = rawPtr(dCenters),
                                            .sizes           = rawPtr(dSizes),
                                            .searchExtFactor = searchExtFactor};

    // split particles into consecutive groups
    constexpr unsigned groupSize = TravConfig::targetSize;
    thrust::universal_vector<LocalIndex> groups((n + groupSize - 1) / groupSize + 1);
    for (unsigned i = 0; i < groups.size(); ++i)
        groups[i] = std::min(groupSize * i, n);
    const GroupView dGroupView{.firstBody  = 0,
                               .lastBody   = n,
                               .numGroups  = unsigned(groups.size() - 1),
                               .groupStart = rawPtr(groups),
                               .groupEnd   = rawPtr(groups) + 1};

    // prefetch vectors to device memory, required on some AMD hardware/software for reasonable performance
    int device;
    checkGpuErrors(cudaGetDevice(&device));
    auto const prefetchToDevice = [&]<class Tv>(const thrust::universal_vector<Tv>& v)
    {
#if defined(__HIPCC__) && (HIP_VERSION_MAJOR < 7 || (HIP_VERSION_MAJOR == 7 && HIP_VERSION_MINOR == 0))
        checkGpuErrors(hipMemPrefetchAsync(rawPtr(v), sizeof(Tv) * v.size(), device));
#else
        checkGpuErrors(cudaMemPrefetchAsync(rawPtr(v), sizeof(Tv) * v.size(), {cudaMemLocationTypeDevice, device}, 0));
#endif
    };
    util::for_each_tuple(prefetchToDevice, std::tie(dX, dY, dZ, dH));
    util::for_each_tuple(prefetchToDevice, dInputs);
    util::for_each_tuple(prefetchToDevice, dOutputs);
    util::for_each_tuple(prefetchToDevice, std::tie(dPrefixes, dChildOffsets, dParents, dInternalToLeaf,
                                                    dLeafToInternal, dLevelRange, dLayout, dCenters, dSizes, groups));
    checkGpuErrors(cudaDeviceSynchronize());

    // build neighborhood, measure CPU time
    using Clock     = std::chrono::high_resolution_clock;
    auto buildStart = Clock::now();
    const auto neighborhood =
        neighborhoodBuilder.build(dNsView, box, n, dGroupView, rawPtr(dX), rawPtr(dY), rawPtr(dZ), hVal);
    checkGpuErrors(cudaDeviceSynchronize());
    auto buildEnd         = Clock::now();
    const float buildTime = std::chrono::duration<float>(buildEnd - buildStart).count();
    printf("Neighborhood build time (CPU time): %7.6f s\n", buildTime);
    const ijloop::Statistics stats  = neighborhood.stats();
    const float numBytesPerParticle = static_cast<float>(stats.numBytes / double(stats.numBodies));
    printf("Memory usage of neighborhood data: %.2f MB (%.1f B/particle)\n", stats.numBytes / 1.0e6,
           numBytesPerParticle);

    // run the actual interaction kernel and measure time with GPU timers
    std::vector<float> times(101);
    std::vector<cudaEvent_t> events(times.size() + 1);
    for (auto& event : events)
        checkGpuErrors(cudaEventCreate(&event));
    checkGpuErrors(cudaEventRecord(events[0]));
    for (std::size_t i = 1; i < events.size(); ++i)
    {
        neighborhood.ijLoop(util::tupleMap([](auto const& v) { return rawPtr(v); }, dInputs),
                            util::tupleMap([](auto& v) { return rawPtr(v); }, dOutputs), interaction,
                            ijloop::empty_postamble);
        checkGpuErrors(cudaEventRecord(events[i]));
    }
    checkGpuErrors(cudaEventSynchronize(events.back()));

    for (std::size_t i = 0; i < times.size(); ++i)
    {
        float millisecs;
        checkGpuErrors(cudaEventElapsedTime(&millisecs, events[i], events[i + 1]));
        checkGpuErrors(cudaEventDestroy(events[i]));
        times[i] = millisecs / 1000.0;
    }
    checkGpuErrors(cudaEventDestroy(events.back()));

    // compute and print mean and standard deviation of performance measurements
    std::vector<double> gigaParticleUpdates(times.size());
    std::transform(times.begin(), times.end(), gigaParticleUpdates.begin(), [&](auto t) { return n / 1.0e9 / t; });

    const float meanTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    const float meanGigaParticleUpdates =
        std::accumulate(gigaParticleUpdates.begin(), gigaParticleUpdates.end(), 0.0f) / gigaParticleUpdates.size();

    const float stdDevTime = std::sqrt(std::accumulate(times.begin(), times.end(), 0.0, [&](auto a, auto t)
                                                       { return a + (t - meanTime) * (t - meanTime); }) /
                                       (times.size() - 1));
    const float stdDevGigaParticleUpdates =
        std::sqrt(std::accumulate(gigaParticleUpdates.begin(), gigaParticleUpdates.end(), 0.0, [&](auto a, auto s)
                                  { return a + (s - meanGigaParticleUpdates) * (s - meanGigaParticleUpdates); }) /
                  (gigaParticleUpdates.size() - 1));

    std::sort(times.begin(), times.end());
    std::sort(gigaParticleUpdates.begin(), gigaParticleUpdates.end());

    printf("GPU Time:    %7.6f +- %7.6f, median = %7.6f [s]\n", meanTime, stdDevTime, times[times.size() / 2]);
    printf("Performance: %7.6f +- %7.6f, median = %7.6f [Giga Particle Updates / s]\n", meanGigaParticleUpdates,
           stdDevGigaParticleUpdates, gigaParticleUpdates[gigaParticleUpdates.size() / 2]);

    if (validate)
    {
        unsigned long numFails = 0;
        const auto isClose     = [](T a, T b)
        {
            if constexpr (std::is_integral_v<T>) { return a == b; }
            else
            {
                constexpr bool isDouble = std::is_same_v<T, double>;
                constexpr double atol   = isDouble ? 1e-6 : 1e-5;
                constexpr double rtol   = isDouble ? 1e-5 : 1e-4;
                return std::abs(a - b) <= atol + rtol * std::abs(b);
            }
        };
        util::for_each_tuple(
            [&](auto const& dOut, auto const& out)
            {
                assert(dOut.size() == n && out.size() == n);
#pragma omp parallel for
                for (unsigned i = 0; i < n; ++i)
                {
                    if (!isClose(dOut[i], out[i]))
                    {
                        unsigned long failNum;
#pragma omp atomic capture
                        failNum = numFails++;
                        if (failNum < 10)
                        {
                            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(dOut[i])>, unsigned>)
                            {
#pragma omp critical
                                printf("FAIL %u: %10u != %10u\n", i, dOut[i], out[i]);
                            }
                            else
                            {
#pragma omp critical
                                printf("FAIL %u: %.10f != %.10f\n", i, dOut[i], out[i]);
                            }
                        }
                    }
                }
            },
            dOutputs, outputs);
        if (numFails) printf("TOTAL FAILS: %lu\n", numFails);
    }

    return {times, buildTime, numBytesPerParticle};
}
