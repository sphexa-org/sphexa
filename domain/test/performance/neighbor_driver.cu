/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Find neighbors in Morton code sorted x,y,z arrays
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <iomanip>
#include <iostream>
#include <iterator>

#include <thrust/device_vector.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/findneighbors.hpp"

#include "cstone/traversal/find_neighbors.cuh"

#include "../coord_samples/random.hpp"
#include "timing.cuh"

using namespace cstone;

//! @brief depth-first traversal based neighbor search
template<class T, class KeyType>
__global__ void findNeighborsKernel(const T* x,
                                    const T* y,
                                    const T* z,
                                    const T* h,
                                    LocalIndex firstId,
                                    LocalIndex lastId,
                                    const Box<T> box,
                                    const OctreeNsView<T, KeyType> treeView,
                                    unsigned ngmax,
                                    LocalIndex* neighbors,
                                    unsigned* neighborsCount)
{
    LocalIndex tid = blockDim.x * blockIdx.x + threadIdx.x;
    LocalIndex id  = firstId + tid;
    if (id >= lastId) { return; }

    neighborsCount[id] = findNeighbors(id, x, y, z, h, treeView, box, ngmax, neighbors + tid * ngmax);
}

/*! @brief Neighbor search for bodies within the specified range
 *
 * @param[in]    firstBody    index of first body in @p bodyPos to compute acceleration for
 * @param[in]    lastBody     index (exclusive) of last body in @p bodyPos to compute acceleration for
 * @param[in]    x,y,z,h      bodies, in SFC order and as referenced by @p layout
 * @param[in]    tree         octree data for traversal
 * @param[in]    box          global coordinate bounding box
 * @param[out]   nc           neighbor counts of bodies with indices in [firstBody, lastBody]
 * @param[-]     globalPool   temporary storage for the cell traversal stack, uninitialized
 *                            each active warp needs space for TravConfig::memPerWarp int32,
 *                            so the total size is TravConfig::memPerWarp * numWarpsPerBlock * numBlocks
 */
template<class Tc, class Th, class KeyType>
__global__ __launch_bounds__(TravConfig::numThreads) void traverseBT(LocalIndex firstBody,
                                                                     LocalIndex lastBody,
                                                                     const Tc* __restrict__ x,
                                                                     const Tc* __restrict__ y,
                                                                     const Tc* __restrict__ z,
                                                                     const Th* __restrict__ h,
                                                                     OctreeNsView<Tc, KeyType> tree,
                                                                     const Box<Tc> box,
                                                                     unsigned* nc,
                                                                     unsigned* nidx,
                                                                     unsigned ngmax,
                                                                     int* globalPool)
{
    const unsigned laneIdx    = threadIdx.x & (GpuConfig::warpSize - 1);
    const unsigned numTargets = (lastBody - firstBody - 1) / TravConfig::targetSize + 1;
    int targetIdx             = 0;

    while (true)
    {
        // first thread in warp grabs next target
        if (laneIdx == 0) { targetIdx = atomicAdd(&targetCounterGlob, 1); }
        targetIdx = shflSync(targetIdx, 0);

        if (targetIdx >= numTargets) return;

        const LocalIndex bodyBegin = firstBody + targetIdx * TravConfig::targetSize;
        const LocalIndex bodyEnd   = imin(bodyBegin + TravConfig::targetSize, lastBody);
        unsigned* warpNidx         = nidx + targetIdx * TravConfig::targetSize * ngmax;

        auto nc_i = traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, warpNidx, ngmax, globalPool);

        const LocalIndex bodyIdxLane = bodyBegin + laneIdx;
        for (int i = 0; i < TravConfig::nwt; i++)
        {
            const LocalIndex bodyIdx = bodyIdxLane + i * GpuConfig::warpSize;
            if (bodyIdx < bodyEnd) { nc[bodyIdx] = nc_i[i]; }
        }
    }
}

template<class Tc, class Th, class KeyType>
auto findNeighborsBT(size_t firstBody,
                     size_t lastBody,
                     const Tc* x,
                     const Tc* y,
                     const Tc* z,
                     const Th* h,
                     OctreeNsView<Tc, KeyType> tree,
                     const Box<Tc>& box,
                     unsigned* nc,
                     unsigned* nidx,
                     unsigned ngmax)
{
    unsigned numBodies = lastBody - firstBody;
    unsigned numBlocks = TravConfig::numBlocks();
    unsigned poolSize  = TravConfig::poolSize();
    thrust::device_vector<int> globalPool(poolSize);

    printf("launching %d blocks\n", numBlocks);
    resetTraversalCounters<<<1, 1>>>();
    auto t0 = std::chrono::high_resolution_clock::now();
    traverseBT<<<numBlocks, TravConfig::numThreads>>>(firstBody, lastBody, x, y, z, h, tree, box, nc, nidx, ngmax,
                                                      rawPtr(globalPool));
    kernelSuccess("traverseBT");

    auto t1   = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();

    NcStats::type stats[NcStats::numStats];
    checkGpuErrors(cudaMemcpyFromSymbol(stats, ncStats, NcStats::numStats * sizeof(uint64_t)));

    NcStats::type sumP2P   = stats[NcStats::sumP2P];
    NcStats::type maxP2P   = stats[NcStats::maxP2P];
    NcStats::type maxStack = stats[NcStats::maxStack];

    util::array<Tc, 2> interactions;
    interactions[0] = Tc(sumP2P) / Tc(numBodies);
    interactions[1] = Tc(maxP2P);

    fprintf(stdout, "Traverse : %.7f s (%.7f TFlops) P2P %f, maxP2P %f, maxStack %llu\n", dt, 11.0 * sumP2P / dt / 1e12,
            interactions[0], interactions[1], maxStack);

    return interactions;
}

template<class T, class StrongKeyType>
void benchmarkGpu()
{
    using KeyType = typename StrongKeyType::ValueType;

    Box<T> box{0, 1, BoundaryType::periodic};
    int n = 2000000;

    RandomCoordinates<T, StrongKeyType> coords(n, box);
    std::vector<T> h(n, 0.012);

    // RandomGaussianCoordinates<T, StrongKeyType> coords(n, box);
    // adjustSmoothingLength<KeyType>(n, 100, 200, coords.x(), coords.y(), coords.z(), h, box);

    int ngmax = 200;

    std::vector<LocalIndex> neighborsCPU(ngmax * n);
    std::vector<unsigned> neighborsCountCPU(n);

    const T* x = coords.x().data();
    const T* y = coords.y().data();
    const T* z = coords.z().data();

    unsigned bucketSize   = 64;
    auto [csTree, counts] = computeOctree(std::span(coords.particleKeys()), bucketSize);
    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(csTree));
    updateInternalTree<KeyType>(csTree, octree.data());

    std::vector<LocalIndex> layout(nNodes(csTree) + 1, 0);
    std::inclusive_scan(counts.begin(), counts.end(), layout.begin() + 1);

    std::vector<Vec3<T>> centers(octree.numNodes), sizes(octree.numNodes);
    std::span<const KeyType> nodeKeys(octree.prefixes.data(), octree.numNodes);
    nodeFpCenters<KeyType>(nodeKeys, centers.data(), sizes.data(), box);

    OctreeNsView<T, KeyType> nsView{octree.numLeafNodes,
                                    octree.prefixes.data(),
                                    octree.childOffsets.data(),
                                    octree.parents.data(),
                                    octree.internalToLeaf.data(),
                                    octree.levelRange.data(),
                                    nullptr,
                                    layout.data(),
                                    centers.data(),
                                    sizes.data()};

    auto findNeighborsCpu = [&]()
    {
#pragma omp parallel for
        for (LocalIndex i = 0; i < n; ++i)
        {
            neighborsCountCPU[i] =
                findNeighbors(i, x, y, z, h.data(), nsView, box, ngmax, neighborsCPU.data() + i * ngmax);
        }
    };

    float cpuTime = timeCpu(findNeighborsCpu);

    std::cout << "CPU time " << cpuTime << " s" << std::endl;
    std::copy(neighborsCountCPU.data(), neighborsCountCPU.data() + 64, std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    std::vector<LocalIndex> neighborsGPU(ngmax * n);
    std::vector<unsigned> neighborsCountGPU(n);

    thrust::device_vector<T> d_x(coords.x().begin(), coords.x().end());
    thrust::device_vector<T> d_y(coords.y().begin(), coords.y().end());
    thrust::device_vector<T> d_z(coords.z().begin(), coords.z().end());
    thrust::device_vector<T> d_h = h;

    thrust::device_vector<KeyType> d_prefixes             = octree.prefixes;
    thrust::device_vector<TreeNodeIndex> d_childOffsets   = octree.childOffsets;
    thrust::device_vector<TreeNodeIndex> d_parents        = octree.parents;
    thrust::device_vector<TreeNodeIndex> d_internalToLeaf = octree.internalToLeaf;
    thrust::device_vector<TreeNodeIndex> d_levelRange     = octree.levelRange;
    thrust::device_vector<LocalIndex> d_layout            = layout;
    thrust::device_vector<Vec3<T>> d_centers              = centers;
    thrust::device_vector<Vec3<T>> d_sizes                = sizes;

    OctreeNsView<T, KeyType> nsViewGpu{octree.numLeafNodes,
                                       rawPtr(d_prefixes),
                                       rawPtr(d_childOffsets),
                                       rawPtr(d_parents),
                                       rawPtr(d_internalToLeaf),
                                       rawPtr(d_levelRange),
                                       nullptr,
                                       rawPtr(d_layout),
                                       rawPtr(d_centers),
                                       rawPtr(d_sizes)};

    thrust::device_vector<LocalIndex> d_neighbors(neighborsGPU.size());
    thrust::device_vector<unsigned> d_neighborsCount(neighborsCountGPU.size());

    auto findNeighborsLambda = [&]()
    {
        // findNeighborsKernel<<<iceil(n, 128), 128>>>(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_h), 0, n, box,
        //                                             nsViewGpu, ngmax, rawPtr(d_neighbors), rawPtr(d_neighborsCount));

        findNeighborsBT(0, n, rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_h), nsViewGpu, box,
                        rawPtr(d_neighborsCount), rawPtr(d_neighbors), ngmax);
    };

    float gpuTime = timeGpu(findNeighborsLambda);

    thrust::copy(d_neighborsCount.begin(), d_neighborsCount.end(), neighborsCountGPU.begin());
    thrust::copy(d_neighbors.begin(), d_neighbors.end(), neighborsGPU.begin());

    std::cout << "GPU time " << gpuTime / 1000 << " s" << std::endl;
    std::copy(neighborsCountGPU.data(), neighborsCountGPU.data() + 64, std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    int numFails     = 0;
    int numFailsList = 0;
    for (int i = 0; i < n; ++i)
    {
        std::sort(neighborsCPU.data() + i * ngmax, neighborsCPU.data() + i * ngmax + neighborsCountCPU[i]);

        std::vector<cstone::LocalIndex> nilist(neighborsCountGPU[i]);
        for (unsigned j = 0; j < neighborsCountGPU[i]; ++j)
        {
            size_t warpOffset = (i / TravConfig::targetSize) * TravConfig::targetSize * ngmax;
            size_t laneOffset = i % TravConfig::targetSize;
            nilist[j]         = neighborsGPU[warpOffset + TravConfig::targetSize * j + laneOffset];
            nilist[j]         = neighborsGPU[warpOffset + TravConfig::targetSize * j + laneOffset];

            // nilist[j] = neighborsGPU[i * ngmax + j];
        }
        std::sort(nilist.begin(), nilist.end());

        if (neighborsCountGPU[i] != neighborsCountCPU[i])
        {
            std::cout << i << " " << neighborsCountGPU[i] << " " << neighborsCountCPU[i] << std::endl;
            numFails++;
        }

        if (!std::equal(begin(nilist), end(nilist), neighborsCPU.begin() + i * ngmax)) { numFailsList++; }
    }

    bool allEqual = std::equal(begin(neighborsCountGPU), end(neighborsCountGPU), begin(neighborsCountCPU));
    if (allEqual)
        std::cout << "Neighbor counts: PASS\n";
    else
        std::cout << "Neighbor counts: FAIL " << numFails << std::endl;

    std::cout << "numFailsList " << numFailsList << std::endl;
}

int main() { benchmarkGpu<double, HilbertKey<uint64_t>>(); }
