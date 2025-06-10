/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Cornerstone octree GPU testing
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cstone/cuda/thrust_util.cuh"
#include "cstone/focus/source_center.hpp"
#include "cstone/traversal/collisions_gpu.h"
#include "cstone/tree/cs_util.hpp"
#include "cstone/tree/octree_gpu.h"

using namespace cstone;

TEST(Macs, limitSource4x4_matchCPU)
{
    using KeyType = uint64_t;
    using T       = double;

    Box<T> box(0, 1);
    float invTheta = sqrt(3.) / 2;

    thrust::device_vector<KeyType> leaves = makeUniformNLevelTree<KeyType>(64, 1);
    OctreeData<KeyType, GpuTag> fullTree;
    fullTree.resize(nNodes(leaves));
    buildOctreeGpu(rawPtr(leaves), fullTree.data());
    OctreeView<KeyType> ov = fullTree.data();

    std::vector<KeyType> h_prefixes = toHost(fullTree.prefixes);
    std::vector<SourceCenterType<T>> h_centers(ov.numNodes);
    geoMacSpheres<KeyType>(h_prefixes, h_centers.data(), invTheta, box);
    thrust::device_vector<uint8_t> macs(ov.numNodes, 0);
    thrust::device_vector<SourceCenterType<T>> centers = h_centers;

    markMacsGpu(ov.prefixes, ov.childOffsets, ov.parents, rawPtr(centers), box, rawPtr(leaves) + 0, 32, true,
                rawPtr(macs));
    thrust::host_vector<uint8_t> h_macs = macs;

    thrust::host_vector<uint8_t> macRef = std::vector<uint8_t>{1, 0, 0, 0, 0, 1, 1, 1, 1};
    macRef.resize(ov.numNodes);
    EXPECT_EQ(macRef, h_macs);

    thrust::fill(macs.begin(), macs.end(), 0);
    markMacsGpu(ov.prefixes, ov.childOffsets, ov.parents, rawPtr(centers), box, rawPtr(leaves) + 0, 32, false,
                rawPtr(macs));
    h_macs      = macs;
    int numMacs = std::accumulate(h_macs.begin(), h_macs.end(), 0);
    EXPECT_EQ(numMacs, 5 + 16);
}
