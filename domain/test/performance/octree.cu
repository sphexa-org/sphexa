/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Benchmark cornerstone octree generation on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "cstone/cuda/thrust_util.cuh"
#include "cstone/traversal/collisions_gpu.h"
#include "cstone/tree/update_gpu.cuh"
#include "cstone/tree/octree_gpu.h"

#include "coord_samples/random.hpp"

#include "timing.cuh"

using namespace cstone;

template<class T, class KeyType>
auto benchmarkMacsCpu(const OctreeView<KeyType>& octree,
                      const SourceCenterType<T>* centers,
                      const Box<T>& box,
                      const thrust::host_vector<KeyType>& leaves,
                      TreeNodeIndex firstFocusNode,
                      TreeNodeIndex lastFocusNode)
{
    std::vector<uint8_t> macs(octree.numNodes, 0);
    auto findMacsLambda = [&octree, &centers, &box, &leaves, &macs, firstFocusNode, lastFocusNode]()
    {
        markMacs(octree.prefixes, octree.childOffsets, centers, box, leaves.data() + firstFocusNode,
                 lastFocusNode - firstFocusNode, false, macs.data());
    };

    float macCpuTime = timeGpu(findMacsLambda);
    std::cout << "CPU mac eval " << macCpuTime / 1000 << " nNodes(tree): " << nNodes(leaves)
              << " count: " << std::accumulate(macs.begin(), macs.end(), 0) << std::endl;
    return macs;
}

int main()
{
    using KeyType = uint64_t;
    using T       = double;
    Box<T> box{-1, 1};

    unsigned numParticles = 2000000;
    unsigned bucketSize   = 16;

    RandomGaussianCoordinates<T, MortonKey<KeyType>> randomBox(numParticles, box);

    thrust::device_vector<KeyType> tree    = std::vector<KeyType>{0, nodeRange<KeyType>(0)};
    thrust::device_vector<unsigned> counts = std::vector<unsigned>{numParticles};

    thrust::device_vector<KeyType> tmpTree;
    thrust::device_vector<TreeNodeIndex> workArray;

    thrust::device_vector<KeyType> particleCodes(randomBox.particleKeys().begin(), randomBox.particleKeys().end());

    // cornerstone build benchmark

    auto fullBuild = [&]()
    {
        while (!updateOctreeGpu<KeyType>({rawPtr(particleCodes), numParticles}, bucketSize, tree, counts, tmpTree,
                                         workArray))
            ;
    };

    float buildTime = timeGpu(fullBuild);
    std::cout << "build time from scratch " << buildTime / 1000 << " nNodes(tree): " << nNodes(tree)
              << " count: " << thrust::reduce(counts.begin(), counts.end(), 0) << std::endl;

    auto updateTree = [&]() {
        updateOctreeGpu<KeyType>({rawPtr(particleCodes), numParticles}, bucketSize, tree, counts, tmpTree, workArray);
    };

    float updateTime = timeGpu(updateTree);
    std::cout << "build time with guess " << updateTime / 1000 << " nNodes(tree): " << nNodes(tree)
              << " count: " << thrust::reduce(counts.begin(), counts.end(), 0) << std::endl;

    // internal tree benchmark

    OctreeData<KeyType, GpuTag> octree;
    octree.resize(nNodes(tree));
    auto buildInternal = [&]() { buildOctreeGpu(rawPtr(tree), octree.data()); };

    float internalBuildTime           = timeGpu(buildInternal);
    std::vector<TreeNodeIndex> ranges = toHost(octree.levelRange);
    std::cout << "internal build time " << internalBuildTime / 1000 << std::endl;
    std::cout << "level ranges: ";
    for (int i = 0; i <= maxTreeLevel<KeyType>{}; ++i)
        std::cout << ranges[i] << " ";
    std::cout << std::endl;

    // halo discovery benchmark

    thrust::device_vector<float> haloRadii(nNodes(tree), 0.01);
    thrust::device_vector<int> flags(nNodes(tree), 0);

    auto octreeView      = octree.data();
    auto findHalosLambda = [octree = octreeView, &box, &tree, &haloRadii, &flags]()
    {
        findHalosGpu(octree.prefixes, octree.childOffsets, octree.internalToLeaf, rawPtr(tree), rawPtr(haloRadii), box,
                     0, octree.numLeafNodes / 4, rawPtr(flags));
    };

    float findTime = timeGpu(findHalosLambda);
    std::cout << "halo discovery " << findTime / 1000 << " nNodes(tree): " << nNodes(tree)
              << " count: " << thrust::reduce(flags.begin(), flags.end(), 0) << std::endl;

    thrust::host_vector<KeyType> h_tree = tree;
    Octree<KeyType> h_octreeHarness;
    h_octreeHarness.update(h_tree.data(), nNodes(h_tree));
    OctreeView<KeyType> h_octree = h_octreeHarness.data();
    {
        thrust::host_vector<float> radii = haloRadii;
        std::vector<int> h_flags(nNodes(tree), 0);

        auto findHalosCpuLambda = [&]()
        {
            findHalos(h_octree.prefixes, h_octree.childOffsets, h_octree.internalToLeaf, h_tree.data(), radii.data(),
                      box, 0, nNodes(tree) / 4, h_flags.data());
        };
        float findTimeCpu = timeCpu(findHalosCpuLambda);
        std::cout << "CPU halo discovery " << findTimeCpu << " nNodes(tree): " << nNodes(h_tree)
                  << " count: " << thrust::reduce(h_flags.begin(), h_flags.end(), 0) << std::endl;
    }

    /*****************************************************/
    //! MAC tests
    /*****************************************************/

    TreeNodeIndex firstFocusNode = 10000 + 0;
    TreeNodeIndex lastFocusNode  = 10000 + octree.numLeafNodes / 2;

    thrust::device_vector<uint8_t> macs(octree.numNodes);
    thrust::device_vector<SourceCenterType<T>> centers(octree.numNodes);

    float invTheta = 1.0 / 0.5;
    std::vector<SourceCenterType<double>> h_centers(octree.numNodes);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < h_octree.numNodes; ++i)
    {
        KeyType prefix   = h_octree.prefixes[i];
        KeyType startKey = decodePlaceholderBit(prefix);
        unsigned level   = decodePrefixLength(prefix) / 3;
        auto nodeBox     = sfcIBox(sfcKey(startKey), level);
        Vec3<T> center_i;
        util::tie(center_i, std::ignore) = centerAndSize<KeyType>(nodeBox, box);

        h_centers[i][0] = center_i[0];
        h_centers[i][1] = center_i[1];
        h_centers[i][2] = center_i[2];
        h_centers[i][3] = computeVecMacR2(h_octree.prefixes[i], center_i, invTheta, box);
    }
    centers = h_centers;

    auto findMacsLambda = [octree = octreeView, &centers, &box, &tree, &macs, firstFocusNode, lastFocusNode]()
    {
        markMacsGpu(octree.prefixes, octree.childOffsets, rawPtr(centers), box, rawPtr(tree) + firstFocusNode,
                    lastFocusNode - firstFocusNode, false, rawPtr(macs));
    };

    float macTime = timeGpu(findMacsLambda);
    std::cout << "mac eval " << macTime / 1000 << " nNodes(tree): " << nNodes(tree)
              << " count: " << thrust::reduce(macs.begin(), macs.end(), 0) << std::endl;

    auto macsCpu = benchmarkMacsCpu(h_octree, h_centers.data(), box, h_tree, firstFocusNode, lastFocusNode);

    thrust::host_vector<uint8_t> macsGpuDl = macs;
    std::cout << "GPU matches CPU " << std::equal(macsCpu.begin(), macsCpu.end(), macsGpuDl.begin());
}
