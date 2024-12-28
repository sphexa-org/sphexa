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

#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "coord_samples/random.hpp"
#include "cstone/cuda/device_vector.h"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/tree/csarray.hpp"
#include "cstone/tree/csarray_gpu.h"
#include "cstone/tree/cs_util.hpp"
#include "cstone/tree/update_gpu.cuh"

using namespace cstone;

TEST(CsArrayGpu, computeNodeCountsGpu)
{
    using KeyType = unsigned;

    // regular level-3 cornerstone tree with 512 leaves
    thrust::host_vector<KeyType> h_cstree = makeUniformNLevelTree<KeyType>(8 * 8 * 8, 1);
    // subdivide the first level-3 node
    for (int octant = 1; octant < 8; ++octant)
    {
        h_cstree.push_back(octant * nodeRange<KeyType>(4));
    }

    std::sort(begin(h_cstree), end(h_cstree));

    // create + upload tree to the device
    thrust::device_vector<KeyType> d_cstree = h_cstree;

    thrust::host_vector<KeyType> h_particleKeys;
    for (int nodeIdx = 1; nodeIdx < nNodes(h_cstree) - 1; ++nodeIdx)
    {
        // put 2 particles in each tree node, except the first and last node
        h_particleKeys.push_back(h_cstree[nodeIdx]);
        h_particleKeys.push_back(h_cstree[nodeIdx] + 1);
    }

    // upload particle codes to device
    thrust::device_vector<KeyType> d_particleKeys = h_particleKeys;

    thrust::device_vector<unsigned> d_counts(nNodes(d_cstree), 1);

    thrust::host_vector<unsigned> refCounts(nNodes(d_cstree), 2);
    // first and last nodes are empty
    refCounts[0]        = 0;
    *refCounts.rbegin() = 0;

    computeNodeCountsGpu(rawPtr(d_cstree), rawPtr(d_counts), nNodes(d_cstree),
                         {rawPtr(d_particleKeys), d_particleKeys.size()}, std::numeric_limits<unsigned>::max(), false);
    thrust::host_vector<unsigned> h_counts = d_counts;
    EXPECT_EQ(h_counts, refCounts);

    // check again, using previous counts as guesses
    computeNodeCountsGpu(rawPtr(d_cstree), rawPtr(d_counts), nNodes(d_cstree),
                         {rawPtr(d_particleKeys), d_particleKeys.size()}, std::numeric_limits<unsigned>::max(), true);
    h_counts = d_counts;
    EXPECT_EQ(h_counts, refCounts);
}

TEST(CsArrayGpu, rebalanceDecision)
{
    using KeyType       = unsigned;
    unsigned bucketSize = 8;

    thrust::device_vector<KeyType> tree = OctreeMaker<KeyType>{}.divide().divide(7).makeTree();
    thrust::device_vector<unsigned> counts(nNodes(tree), 1);
    counts[1] = 9;
    thrust::fill_n(counts.begin() + 8, 7, 0);

    thrust::device_vector<TreeNodeIndex> nodeOps(tree.size());
    computeNodeOpsGpu(rawPtr(tree), nNodes(tree), rawPtr(counts), bucketSize, rawPtr(nodeOps));

    // regular level-3 cornerstone tree with 512 leaves
    thrust::host_vector<TreeNodeIndex> h_nodeOps = nodeOps;

    thrust::host_vector<TreeNodeIndex> refNodeOps =
        std::vector<TreeNodeIndex>{0, 1, 9, 10, 11, 12, 13, 14, 15, 15, 15, 15, 15, 15, 15, 15};

    EXPECT_EQ(refNodeOps, h_nodeOps);
}

TEST(CsArrayGpu, rebalanceTree)
{
    using KeyType                       = unsigned;
    thrust::device_vector<KeyType> tree = OctreeMaker<KeyType>{}.divide().divide(7).makeTree();

    // node {1} to be split, nodes {7,i} are to be fused
    thrust::device_vector<TreeNodeIndex> nodeOps =
        std::vector<TreeNodeIndex>{0, 1, 9, 10, 11, 12, 13, 14, 15, 15, 15, 15, 15, 15, 15, 15};
    thrust::device_vector<KeyType> newTree(*nodeOps.rbegin() + 1);

    bool converged = rebalanceTreeGpu(rawPtr(tree), nNodes(tree), nNodes(newTree), rawPtr(nodeOps), rawPtr(newTree));

    // download tree from host
    thrust::host_vector<KeyType> h_tree    = newTree;
    thrust::host_vector<KeyType> reference = OctreeMaker<KeyType>{}.divide().divide(1).makeTree();
    EXPECT_EQ(h_tree, reference);
    EXPECT_FALSE(converged);
}

/*! @brief fixture for octree tests based on random particle distributions
 *
 * @tparam KeyType         32- or 64-bit unsigned integer
 *
 * These tests are already integration tests strictly speaking. They can be seen
 * as the second line of defense in case the unit tests above (with minimal and explict reference data)
 * fail to catch an error.
 */
template<class KeyType>
class OctreeFixtureGpu
{
public:
    OctreeFixtureGpu(unsigned numParticles, unsigned bucketSize)
    {
        d_codes = makeRandomGaussianKeys<KeyType>(numParticles);

        d_tree   = std::vector<KeyType>{0, nodeRange<KeyType>(0)};
        d_counts = std::vector<unsigned>{numParticles};

        DeviceVector<KeyType> tmpTree;
        DeviceVector<TreeNodeIndex> workArray;

        while (!updateOctreeGpu<KeyType>({rawPtr(d_codes), d_codes.size()}, bucketSize, d_tree, d_counts, tmpTree,
                                         workArray))
            ;
    }

    DeviceVector<KeyType> d_tree;
    DeviceVector<KeyType> d_codes;
    DeviceVector<unsigned> d_counts;
};

//! @brief build tree from random particles and compare against CPU
TEST(CsArrayGpu, computeOctreeRandom)
{
    using KeyType = unsigned;

    int nParticles = 100000;
    int bucketSize = 64;

    // compute octree starting from default uniform octree
    auto particleKeys         = makeRandomGaussianKeys<KeyType>(nParticles);
    auto [treeCpu, countsCpu] = computeOctree<KeyType>(particleKeys, bucketSize);

    OctreeFixtureGpu<KeyType> fixt(nParticles, bucketSize);

    // upload CPU reference to GPU
    DeviceVector<KeyType> refTreeCpu    = treeCpu;
    DeviceVector<unsigned> refCountsCpu = countsCpu;

    EXPECT_EQ(fixt.d_tree, refTreeCpu);
    EXPECT_EQ(fixt.d_counts, refCountsCpu);
}

/*! @brief simulation of distributed tree
 *
 * In distributed octrees, the executing rank only has a part of the particle SFC codes, such that
 * many nodes in the tree are empty. Here this is simulated by removing a large connected part of the particle codes
 * and recomputing the node counts based on this subset of particle codes. The non-zero node counts should stay the
 * same.
 */
TEST(CsArrayGpu, distributedMockUp)
{
    using CodeType = unsigned;

    int nParticles = 100000;
    int bucketSize = 64;

    OctreeFixtureGpu<CodeType> fixt(nParticles, bucketSize);

    DeviceVector<CodeType> d_counts_orig = fixt.d_counts;

    // omit first and last tenth of nodes
    TreeNodeIndex Nodes     = nNodes(fixt.d_tree);
    TreeNodeIndex firstNode = Nodes / 10;
    TreeNodeIndex lastNode  = Nodes - Nodes / 10;

    // determine the part of the tree that will be empty
    CodeType nodeKey1, nodeKey2;
    memcpyD2H(fixt.d_tree.data() + firstNode, 1, &nodeKey1);
    memcpyD2H(fixt.d_tree.data() + lastNode, 1, &nodeKey2);
    unsigned firstIdx = lowerBoundGpu(fixt.d_codes.data(), fixt.d_codes.data() + fixt.d_codes.size(), nodeKey1);
    unsigned lastIdx  = lowerBoundGpu(fixt.d_codes.data(), fixt.d_codes.data() + fixt.d_codes.size(), nodeKey2);
    std::cout << firstNode << " " << lastNode << std::endl;
    std::cout << firstIdx << " " << lastIdx << std::endl;

    bool useCountsAsGuess = true;
    computeNodeCountsGpu(fixt.d_tree.data(), fixt.d_counts.data(), nNodes(fixt.d_tree),
                         {fixt.d_codes.data() + firstIdx, fixt.d_codes.data() + lastIdx},
                         std::numeric_limits<unsigned>::max(), useCountsAsGuess);

    DeviceVector<CodeType> d_counts_ref = d_counts_orig;
    thrust::fill(thrust::device, d_counts_ref.data(), d_counts_ref.data() + firstNode, 0);
    thrust::fill(thrust::device, d_counts_ref.data() + lastNode, d_counts_ref.data() + d_counts_ref.size(), 0);

    EXPECT_EQ(fixt.d_counts, d_counts_ref);
}
