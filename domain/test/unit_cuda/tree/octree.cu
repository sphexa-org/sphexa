/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief internal octree GPU testing
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/tree/octree_gpu.h"
#include "cstone/tree/octree.hpp"
#include "cstone/tree/cs_util.hpp"

using namespace cstone;

template<class KeyType>
void compareAgainstCpu(const std::vector<KeyType>& tree)
{
    // upload cornerstone tree to device
    DeviceVector<KeyType> d_leaves = tree;

    OctreeData<KeyType, GpuTag> gpuTree;
    gpuTree.resize(nNodes(tree));

    buildOctreeGpu(rawPtr(d_leaves), gpuTree.data());

    Octree<KeyType> cpuTree;
    cpuTree.update(tree.data(), nNodes(tree));

    std::vector<KeyType> h_prefixes = toHost(gpuTree.prefixes);
    for (auto& p : h_prefixes)
    {
        p = decodePlaceholderBit(p);
    }

    EXPECT_EQ(cpuTree.numTreeNodes(), gpuTree.numLeafNodes + gpuTree.numInternalNodes);
    for (TreeNodeIndex i = 0; i < cpuTree.numTreeNodes(); ++i)
    {
        EXPECT_EQ(h_prefixes[i], cpuTree.codeStart(i));
    }

    std::vector<TreeNodeIndex> h_children = toHost(gpuTree.childOffsets);
    for (TreeNodeIndex i = 0; i < cpuTree.numTreeNodes(); ++i)
    {
        EXPECT_EQ(h_children[i], cpuTree.child(i, 0));
    }

    std::vector<TreeNodeIndex> h_parents = toHost(gpuTree.parents);
    for (TreeNodeIndex i = 0; i < cpuTree.numTreeNodes(); ++i)
    {
        EXPECT_EQ(h_parents[(i - 1) / 8], cpuTree.parent(i));
    }

    EXPECT_EQ(gpuTree.levelRange.size(), cpuTree.levelRange().size());
    EXPECT_EQ(gpuTree.levelRange.size(), maxTreeLevel<KeyType>{} + 2);
    for (unsigned level = 0; level < gpuTree.levelRange.size(); ++level)
    {
        EXPECT_EQ(gpuTree.levelRange[level], cpuTree.levelRange()[level]);
    }
}

//! @brief This creates an irregular tree. Checks geometry relations between children and parents.
template<class KeyType>
void octreeIrregularL3()
{
    std::vector<KeyType> tree = OctreeMaker<KeyType>{}.divide().divide(0).divide(0, 2).divide(3).makeTree();

    compareAgainstCpu(tree);
}

TEST(OctreeGpu, irregularL3)
{
    octreeIrregularL3<unsigned>();
    octreeIrregularL3<uint64_t>();
}

template<class KeyType>
void octreeRegularL6()
{
    // uniform level-6 tree with 262144
    std::vector<KeyType> tree = makeUniformNLevelTree<KeyType>(64 * 64 * 64, 1);

    compareAgainstCpu(tree);
}

TEST(OctreeGpu, regularL6)
{
    octreeRegularL6<unsigned>();
    octreeRegularL6<uint64_t>();
}
