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

#include "cstone/cuda/device_vector.h"
#include "cstone/tree/cs_util.hpp"
#include "cstone/focus/octree_focus.hpp"

using namespace cstone;

class MacRefinementGpu : public testing::Test
{
protected:
    using KeyType = uint64_t;
    using T       = double;

    void SetUp() override
    {
        OctreeMaker<KeyType> om;
        om.divide().divide(0);
        for (int j = 0; j < 8; ++j)
            om.divide(0, j);
        om.divide(0, 0, 0);

        // L3 in first octant, L1 otherwise
        h_leaves = om.makeTree();
        leaves   = h_leaves;

        octree.resize(nNodes(h_leaves));
        buildOctreeGpu<KeyType>(leaves.data(), octree.data());
        ov = octree.data();
    }

    std::vector<KeyType> h_leaves;
    DeviceVector<KeyType> leaves;
    OctreeData<KeyType, GpuTag> octree;
    OctreeView<KeyType> ov;
    DeviceVector<SourceCenterType<T>> centers;
    DeviceVector<uint8_t> macs;
};

TEST_F(MacRefinementGpu, fullSurface)
{
    Box<T> box(0, 1);
    float invTheta = sqrt(3) / 2 + 1e-6;

    KeyType focusStart = 0;
    KeyType focusEnd   = decodePlaceholderBit(KeyType(011));
    while (!macRefineGpu(octree, leaves, centers, macs, focusEnd, focusEnd, focusStart, focusEnd, invTheta, box)) {}

    int numNodesVertex = 7 + 8;
    int numNodesEdge   = 6 + 2 * 8;
    int numNodesFace   = 4 + 4 * 8;
    EXPECT_EQ(nNodes(leaves), 64 + 7 + 3 * numNodesFace + 3 * numNodesEdge + numNodesVertex);
}

TEST_F(MacRefinementGpu, noSurface)
{
    Box<T> box(0, 1);
    float invTheta              = sqrt(3) / 2 + 1e-6;
    TreeNodeIndex numNodesStart = octree.numLeafNodes;

    KeyType oldFStart  = decodePlaceholderBit(KeyType(0101));
    KeyType oldFEnd    = decodePlaceholderBit(KeyType(011));
    KeyType focusStart = 0;
    KeyType focusEnd   = decodePlaceholderBit(KeyType(011));
    while (!macRefineGpu(octree, leaves, centers, macs, oldFStart, oldFEnd, focusStart, focusEnd, invTheta, box)) {}

    EXPECT_EQ(nNodes(leaves), numNodesStart);
}

TEST_F(MacRefinementGpu, partialSurface)
{
    Box<T> box(0, 1);
    float invTheta              = sqrt(3) / 2 + 1e-6;
    TreeNodeIndex numNodesStart = octree.numLeafNodes;

    KeyType oldFStart  = 0;
    KeyType oldFEnd    = decodePlaceholderBit(KeyType(0107));
    KeyType focusStart = 0;
    KeyType focusEnd   = decodePlaceholderBit(KeyType(011));
    while (!macRefineGpu(octree, leaves, centers, macs, oldFStart, oldFEnd, focusStart, focusEnd, invTheta, box)) {}

    EXPECT_EQ(nNodes(leaves), numNodesStart + 5 * 7);
}