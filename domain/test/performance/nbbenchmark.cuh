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
 * @brief  Common testing infrastructure for ij-loop benchmarks
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <cmath>
#include <cstdio>
#include <map>
#include <numeric>
#include <fstream>
#include <vector>

#include <thrust/universal_vector.h>

#include "cstone/cuda/thrust_util.cuh"
#include "cstone/sfc/box.hpp"
#include "cstone/traversal/ijloop/cpu.hpp"
#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/traversal/groups.hpp"
#include "cstone/traversal/groups_gpu.cuh"
#include "cstone/tree/octree.hpp"
#include "cstone/util/tuple_util.hpp"

template<class Tc,
         class T,
         class StrongKeyType,
         class Coords,
         cstone::ijloop::Neighborhood Neighborhood,
         class Interaction,
         class... InputTs,
         class... OutputTs>
std::vector<double> benchmarkNeighborhood(const Coords& coords,
                                          const Neighborhood& neighborhood,
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
    const std::vector<T> h(n, hVal);
    const Box<Tc> box               = coords.box();
    const double r                  = 2 * hVal;
    const double expected_neighbors = 4.0 / 3.0 * M_PI * r * r * r * n / (box.lx() * box.ly() * box.lz());
    printf("Number of particles: %u\n", n);
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

    constexpr unsigned bucketSize = 64;
    const auto [csTree, counts]   = computeOctree(std::span(coords.particleKeys()), bucketSize);
    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(csTree));
    updateInternalTree<KeyType>(csTree, octree.data());
    const TreeNodeIndex* childOffsets = octree.childOffsets.data();
    const TreeNodeIndex* toLeafOrder  = octree.internalToLeaf.data();

    std::vector<LocalIndex> layout(nNodes(csTree) + 1, 0);
    std::inclusive_scan(counts.begin(), counts.end(), layout.begin() + 1);

    std::vector<Vec3<Tc>> centers(octree.numNodes), sizes(octree.numNodes);
    std::span<const KeyType> nodeKeys(octree.prefixes.data(), octree.numNodes);
    nodeFpCenters<KeyType>(nodeKeys, centers.data(), sizes.data(), box);

    const OctreeNsView<Tc, KeyType> nsView{.numLeafNodes    = octree.numLeafNodes,
                                           .prefixes        = octree.prefixes.data(),
                                           .childOffsets    = octree.childOffsets.data(),
                                           .internalToLeaf  = octree.internalToLeaf.data(),
                                           .levelRange      = octree.levelRange.data(),
                                           .leaves          = keys,
                                           .layout          = layout.data(),
                                           .centers         = centers.data(),
                                           .sizes           = sizes.data(),
                                           .searchExtFactor = 1};
    LocalIndex zero = 0;
    const GroupView groupView{.firstBody = 0, .lastBody = n, .numGroups = 1, .groupStart = &zero, .groupEnd = &n};

    const auto allocVec = [n]<class Tv>(Tv initialValue) { return std::vector<Tv>(n, initialValue); };
    const std::tuple<std::vector<InputTs>...> inputs = util::tupleMap(allocVec, inputValues);
    std::tuple<std::vector<OutputTs>...> outputs     = util::tupleMap(allocVec, initialOutputValues);
    ijloop::CpuAlwaysTraverseNeighborhood{ngmax}
        .build(nsView, box, n, groupView, x, y, z, h.data())
        .ijLoop(util::tupleMap([](auto const& v) { return v.data(); }, inputs),
                util::tupleMap([](auto& v) { return v.data(); }, outputs), interaction, ijloop::empty_postamble);

    const thrust::universal_vector<Tc> dX(coords.x().begin(), coords.x().end()),
        dY(coords.y().begin(), coords.y().end()), dZ(coords.z().begin(), coords.z().end());
    const auto allocGpuVec = [n]<class Tv>(Tv initialValue) { return thrust::universal_vector<Tv>(n, initialValue); };
    const auto dH          = allocGpuVec(hVal);
    const std::tuple<thrust::universal_vector<InputTs>...> dInputs = util::tupleMap(allocGpuVec, inputValues);
    std::tuple<thrust::universal_vector<OutputTs>...> dOutputs     = util::tupleMap(allocGpuVec, initialOutputValues);

    std::size_t particleMemoryUsage = (dX.size() + dY.size() + dZ.size()) * sizeof(Tc);
    const auto addMemoryUsage       = [&]<class Tv>(thrust::universal_vector<Tv> const& v)
    { particleMemoryUsage += v.size() * sizeof(Tv); };
    util::for_each_tuple(addMemoryUsage, dInputs);
    util::for_each_tuple(addMemoryUsage, dOutputs);
    printf("Memory usage of particle data: %.2f MB\n", particleMemoryUsage / 1.0e6);

    const thrust::universal_vector<KeyType> dPrefixes             = octree.prefixes;
    const thrust::universal_vector<TreeNodeIndex> dChildOffsets   = octree.childOffsets;
    const thrust::universal_vector<TreeNodeIndex> dInternalToLeaf = octree.internalToLeaf;
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
                                            .prefixes        = rawPtr(dPrefixes),
                                            .childOffsets    = rawPtr(dChildOffsets),
                                            .internalToLeaf  = rawPtr(dInternalToLeaf),
                                            .levelRange      = rawPtr(dLevelRange),
                                            .leaves          = rawPtr(dCodes),
                                            .layout          = rawPtr(dLayout),
                                            .centers         = rawPtr(dCenters),
                                            .sizes           = rawPtr(dSizes),
                                            .searchExtFactor = searchExtFactor};

    constexpr unsigned groupSize = TravConfig::targetSize;
    DeviceVector<LocalIndex> temp, groups;
    computeGroupSplits(0, n, rawPtr(dX), rawPtr(dY), rawPtr(dZ), rawPtr(dH), dNsView.leaves, dNsView.numLeafNodes,
                       dNsView.layout, box, groupSize, 8, temp, groups);
    const GroupView dGroupView{.firstBody  = 0,
                               .lastBody   = n,
                               .numGroups  = unsigned(groups.size() - 1),
                               .groupStart = rawPtr(groups),
                               .groupEnd   = rawPtr(groups) + 1};
    printf("Number of groups: %u (unsplit: %u)\n", dGroupView.numGroups, (n + groupSize - 1) / groupSize);

    const auto neighborhoodGPU =
        neighborhood.build(dNsView, box, n, dGroupView, rawPtr(dX), rawPtr(dY), rawPtr(dZ), rawPtr(dH));
    const ijloop::Statistics stats = neighborhoodGPU.stats();
    printf("Memory usage of neighborhood data: %.2f MB (%.1f B/particle)\n", stats.numBytes / 1.0e6,
           stats.numBytes / double(stats.numBodies));

    int device;
    checkGpuErrors(cudaGetDevice(&device));
    auto const prefetchToDevice = [&]<class Tv>(const thrust::universal_vector<Tv>& v)
    { checkGpuErrors(cudaMemPrefetchAsync(rawPtr(v), sizeof(Tv) * v.size(), device, 0)); };
    util::for_each_tuple(prefetchToDevice, std::tie(dX, dY, dZ, dH));
    util::for_each_tuple(prefetchToDevice, dInputs);
    util::for_each_tuple(prefetchToDevice, dOutputs);

    std::vector<double> times(101);
    std::vector<cudaEvent_t> events(times.size() + 1);
    for (auto& event : events)
        checkGpuErrors(cudaEventCreate(&event));
    checkGpuErrors(cudaEventRecord(events[0]));
    for (std::size_t i = 1; i < events.size(); ++i)
    {
        neighborhoodGPU.ijLoop(util::tupleMap([](auto const& v) { return rawPtr(v); }, dInputs),
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

    std::vector<double> gigaAtomSteps(times.size());
    std::transform(times.begin(), times.end(), gigaAtomSteps.begin(), [&](auto t) { return n / 1.0e9 / t; });

    const float meanTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    const float meanGigaAtomSteps =
        std::accumulate(gigaAtomSteps.begin(), gigaAtomSteps.end(), 0.0f) / gigaAtomSteps.size();

    const float stdDevTime = std::sqrt(std::accumulate(times.begin(), times.end(), 0.0, [&](auto a, auto t)
                                                       { return a + (t - meanTime) * (t - meanTime); }) /
                                       (times.size() - 1));
    const float stdDevGigaAtomSteps =
        std::sqrt(std::accumulate(gigaAtomSteps.begin(), gigaAtomSteps.end(), 0.0, [&](auto a, auto s)
                                  { return a + (s - meanGigaAtomSteps) * (s - meanGigaAtomSteps); }) /
                  (gigaAtomSteps.size() - 1));

    std::sort(times.begin(), times.end());
    std::sort(gigaAtomSteps.begin(), gigaAtomSteps.end());

    printf("GPU Time:    %7.6f +- %7.6f, median = %7.6f [s]\n", meanTime, stdDevTime, times[times.size() / 2]);
    printf("Performance: %7.6f +- %7.6f, median = %7.6f [Giga Particle Updates / s]\n", meanGigaAtomSteps,
           stdDevGigaAtomSteps, gigaAtomSteps[gigaAtomSteps.size() / 2]);

    if (validate)
    {
        unsigned long numFails = 0;
        const auto isClose     = [](T a, T b)
        {
            constexpr bool isDouble = std::is_same_v<T, double>;
            constexpr double atol   = isDouble ? 1e-6 : 1e-5;
            constexpr double rtol   = isDouble ? 1e-5 : 1e-4;
            return std::abs(a - b) <= atol + rtol * std::abs(b);
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
#pragma omp critical
                            printf("FAIL %u: %.10f != %.10f\n", i, dOut[i], out[i]);
                        }
                    }
                }
            },
            dOutputs, outputs);
        if (numFails) printf("TOTAL FAILS: %lu\n", numFails);
    }

    return times;
}

template<class Path, class T>
void saveCsv(const Path& filename, const std::map<std::string, std::vector<T>>& data)
{
    if (data.empty()) throw std::runtime_error("ERROR writing CSV: no data passed!");
    std::ofstream file(filename);
    if (!file) throw std::runtime_error("ERROR writing CSV: could not open file for writing!");

    std::size_t numRows = 0;
    {
        bool first = true;
        for (const auto& [name, vec] : data)
        {
            if (first)
                first = false;
            else
                file << ",";
            file << name;
            numRows = std::max(numRows, vec.size());
        }
    }
    file << "\n";
    for (std::size_t row = 0; row < numRows; ++row)
    {
        bool first = true;
        for (const auto& [_, vec] : data)
        {
            if (first)
                first = false;
            else
                file << ",";
            if (row < vec.size()) file << vec[row];
        }
        file << "\n";
    }
}
