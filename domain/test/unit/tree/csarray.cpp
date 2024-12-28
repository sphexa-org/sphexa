/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Test cornerstone octree core functionality
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/tree/csarray.hpp"
#include "cstone/tree/cs_util.hpp"

#include "coord_samples/random.hpp"

using namespace cstone;

template<class T>
using pair = util::array<T, 2>;

static void testBounds(LocalIndex guess, uint64_t searchKey, pair<LocalIndex> ref)
{
    using KeyType = uint64_t;
    auto e        = nodeRange<KeyType>(0);
    //                         0   1   2   3   4   5   6   7   8   9 10 11 12
    std::vector<KeyType> codes{3, 10, 11, 14, 16, 16, 16, 18, 19, 21, e, e, e};
    const KeyType* c = codes.data();

    auto x = findSearchBounds(guess, searchKey, c, c + codes.size());
    EXPECT_EQ(x[0], c + ref[0]);
    EXPECT_EQ(x[1], c + ref[1]);
    EXPECT_EQ(stl::lower_bound(c, c + codes.size(), searchKey) - c, stl::lower_bound(x[0], x[1], searchKey) - c);
}

TEST(CornerstoneOctree, searchBounds1) { testBounds(3, 14, {2, 3}); }
TEST(CornerstoneOctree, searchBounds2) { testBounds(3, 15, {3, 4}); }
TEST(CornerstoneOctree, searchBounds3) { testBounds(3, 16, {3, 7}); }
TEST(CornerstoneOctree, searchBounds4) { testBounds(4, 16, {3, 4}); }
TEST(CornerstoneOctree, searchBounds5) { testBounds(0, 17, {0, 8}); }
TEST(CornerstoneOctree, searchBounds6) { testBounds(4, 12, {2, 4}); }
TEST(CornerstoneOctree, searchBounds7) { testBounds(4, 11, {0, 4}); }
TEST(CornerstoneOctree, searchBounds8) { testBounds(4, 10, {0, 4}); }
TEST(CornerstoneOctree, searchBounds9) { testBounds(8, 16, {0, 8}); }
TEST(CornerstoneOctree, searchBounds10) { testBounds(6, 16, {2, 6}); }
TEST(CornerstoneOctree, searchBounds11) { testBounds(9, 21, {8, 9}); }
TEST(CornerstoneOctree, searchBounds12) { testBounds(12, 16, {0, 12}); }
TEST(CornerstoneOctree, searchBounds13) { testBounds(8, nodeRange<uint64_t>(0), {10, 13}); }

//! @brief test that computeNodeCounts correctly counts the number of codes for each node
template<class KeyType>
static void checkCountTreeNodes()
{
    std::vector<KeyType> tree = OctreeMaker<KeyType>{}.divide().divide(0).makeTree();

    std::vector<KeyType> keys{tree[1],         tree[1],       tree[1] + 10, tree[1] + 100, tree[2] - 1,
                              tree[2] + 1,     tree[11],      tree[11] + 2, tree[12],      tree[12] + 1000,
                              tree[12] + 2000, tree[13] - 10, tree[13],     tree[13] + 1};

    //  nodeIdx                     0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
    std::vector<unsigned> reference{0, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 2, 0};
    // code start location             0  5  6  6  6  6  6  6  6  6  6  8 12
    // guess start location            0  1  2  3  4  5  6  7  8  9 10 11 12
    // Ntot: 14, nNonZeroNodes: 13 (first and last node are empty), avgNodeCount: 14/13 = 1

    std::vector<unsigned> counts(nNodes(tree));
    computeNodeCounts<KeyType>(tree.data(), counts.data(), nNodes(tree), keys, std::numeric_limits<unsigned>::max());

    EXPECT_EQ(counts, reference);
}

TEST(CornerstoneOctree, countTreeNodes32) { checkCountTreeNodes<unsigned>(); }

TEST(CornerstoneOctree, countTreeNodes64) { checkCountTreeNodes<uint64_t>(); }

template<class KeyType>
static void computeNodeCountsSTree()
{
    std::vector<KeyType> cornerstones{0, 1, nodeRange<KeyType>(0) - 1, nodeRange<KeyType>(0)};
    std::vector<KeyType> tree = computeSpanningTree<KeyType>(cornerstones);

    /// 2 particles in the first and last node

    /// 2 particles in the first and last node
    std::vector<KeyType> keys{0, 0, nodeRange<KeyType>(0) - 1, nodeRange<KeyType>(0) - 1};

    std::vector<unsigned> countsReference(nNodes(tree), 0);
    countsReference.front() = countsReference.back() = 2;

    std::vector<unsigned> countsProbe(nNodes(tree));
    computeNodeCounts<KeyType>(tree.data(), countsProbe.data(), nNodes(tree), keys,
                               std::numeric_limits<unsigned>::max());
    EXPECT_EQ(countsReference, countsProbe);
}

TEST(CornerstoneOctree, computeNodeCounts_spanningTree)
{
    computeNodeCountsSTree<unsigned>();
    computeNodeCountsSTree<uint64_t>();
}

template<class CodeType, class LocalIndex>
static void rebalanceDecision()
{
    std::vector<CodeType> tree = OctreeMaker<CodeType>{}.divide().divide(0).makeTree();

    unsigned bucketSize = 4;
    std::vector<unsigned> counts{1, 1, 1, 0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 8};

    std::vector<LocalIndex> nodeOps(nNodes(tree));
    bool converged = rebalanceDecision(tree.data(), counts.data(), nNodes(tree), bucketSize, nodeOps.data());

    std::vector<LocalIndex> reference{1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 8, 8, 8, 8};
    EXPECT_EQ(nodeOps, reference);
    EXPECT_FALSE(converged);
}

TEST(CornerstoneOctree, rebalanceDecision)
{
    rebalanceDecision<unsigned, unsigned>();
    rebalanceDecision<uint64_t, unsigned>();
}

template<class CodeType, class LocalIndex>
static void rebalanceDecisionSingleRoot()
{
    std::vector<CodeType> tree = OctreeMaker<CodeType>{}.makeTree();

    unsigned bucketSize = 4;
    std::vector<unsigned> counts{1};

    std::vector<LocalIndex> nodeOps(nNodes(tree));
    bool converged = rebalanceDecision(tree.data(), counts.data(), nNodes(tree), bucketSize, nodeOps.data());

    std::vector<LocalIndex> reference{1};
    EXPECT_EQ(nodeOps, reference);
    EXPECT_TRUE(converged);
}

TEST(CornerstoneOctree, rebalanceDecisionSingleRoot)
{
    rebalanceDecisionSingleRoot<unsigned, unsigned>();
    rebalanceDecisionSingleRoot<uint64_t, unsigned>();
}

/*! @brief test behavior of a maximum-depth tree under rebalancing
 *
 *  Node 0 is at the lowest octree level (10 or 21) and its particle
 *  count is bigger than the bucket size. This test verifies that
 *  this tree stays invariant under rebalancing because the capacity
 *  of the underlying 30-bit or 63 bit Morton code is exhausted.
 */
template<class CodeType>
static void rebalanceInsufficentResolution()
{
    constexpr int bucketSize = 1;

    OctreeMaker<CodeType> octreeMaker;
    for (unsigned level = 0; level < maxTreeLevel<CodeType>{}; ++level)
        octreeMaker.divide({}, level);

    std::vector<CodeType> tree = octreeMaker.makeTree();

    std::vector<unsigned> counts(nNodes(tree), 1);
    counts[0] = bucketSize + 1;

    std::vector<TreeNodeIndex> nodeOps(tree.size());
    // the first node has two particles, one more than the bucketSize
    // since the first node is at the maximum subdivision layer, the tree
    // can't be further refined to satisfy the bucketSize
    bool converged = rebalanceDecision(tree.data(), counts.data(), nNodes(tree), bucketSize, nodeOps.data());

    std::vector<TreeNodeIndex> reference(tree.size(), 1);
    reference[nNodes(tree)] = 0; // last value is for the scan result, irrelevant here

    EXPECT_EQ(nodeOps, reference);
    EXPECT_TRUE(converged);
}

TEST(CornerstoneOctree, rebalanceInsufficientResolution)
{
    rebalanceInsufficentResolution<unsigned>();
    rebalanceInsufficentResolution<uint64_t>();
}

//! @brief check that nodes can be fused at the start of the tree
template<class CodeType>
static void rebalanceTree()
{
    std::vector<CodeType> tree = OctreeMaker<CodeType>{}.divide().divide(0).makeTree();

    std::vector<TreeNodeIndex> nodeOps{1, 0, 0, 0, 0, 0, 0, 0, 1, 8, 1, 1, 1, 1, 8, 0};
    ASSERT_EQ(nodeOps.size(), tree.size());

    std::vector<CodeType> newTree;
    rebalanceTree(tree, newTree, nodeOps.data());

    std::vector<CodeType> reference = OctreeMaker<CodeType>{}.divide().divide(2).divide(7).makeTree();
    EXPECT_EQ(newTree, reference);
}

TEST(CornerstoneOctree, rebalance)
{
    rebalanceTree<unsigned>();
    rebalanceTree<uint64_t>();
}

template<class KeyType>
static void updateTreelet()
{
    unsigned bucketSize = 64;
    auto tree           = OctreeMaker<KeyType>{}.divide().divide(0).makeTree();

    TreeNodeIndex treeletStart = 7;
    TreeNodeIndex treeletEnd   = 9;

    {
        std::span<const KeyType> treelet(tree.data() + treeletStart, tree.data() + treeletEnd + 1);
        std::vector<unsigned> treeletCounts{bucketSize + 1, bucketSize - 1};

        auto newTreelet = updateTreelet<KeyType>(treelet, treeletCounts, bucketSize);

        std::vector<KeyType> reference{treelet.front(),      pad(KeyType(071), 9), pad(KeyType(072), 9),
                                       pad(KeyType(073), 9), pad(KeyType(074), 9), pad(KeyType(075), 9),
                                       pad(KeyType(076), 9), pad(KeyType(077), 9), pad(KeyType(01), 3),
                                       treelet.back()};
        EXPECT_EQ(nNodes(newTreelet), 9);
        EXPECT_EQ(newTreelet, reference);
    }
    {
        std::span<const KeyType> treelet(tree.data() + treeletStart, tree.data() + treeletEnd + 1);
        std::vector<unsigned> treeletCounts{bucketSize - 1, bucketSize + 1};

        auto newTreelet = updateTreelet<KeyType>(treelet, treeletCounts, bucketSize);

        std::vector<KeyType> reference{treelet.front(),      pad(KeyType(010), 6), pad(KeyType(011), 6),
                                       pad(KeyType(012), 6), pad(KeyType(013), 6), pad(KeyType(014), 6),
                                       pad(KeyType(015), 6), pad(KeyType(016), 6), pad(KeyType(017), 6),
                                       treelet.back()};
        EXPECT_EQ(nNodes(newTreelet), 9);
        EXPECT_EQ(newTreelet, reference);
    }
    treeletStart = 0;
    treeletEnd   = 8;
    {
        std::span<const KeyType> treelet(tree.data() + treeletStart, tree.data() + treeletEnd + 1);
        std::vector<unsigned> treeletCounts{1, 2, 3, 4, 5, 6, 7, 8};

        auto newTreelet = updateTreelet<KeyType>(treelet, treeletCounts, bucketSize);

        std::vector<KeyType> reference{treelet.front(), treelet.back()};
        EXPECT_EQ(nNodes(newTreelet), 1);
        EXPECT_EQ(newTreelet, reference);
    }
}

TEST(CornerstoneOctree, updateTreelet)
{
    updateTreelet<unsigned>();
    updateTreelet<uint64_t>();
}

template<class KeyType>
static void checkOctreeWithCounts(const std::vector<KeyType>& tree,
                                  const std::vector<unsigned>& counts,
                                  int bucketSize,
                                  const std::vector<KeyType>& mortonCodes,
                                  bool relaxBucketCount)
{
    using CodeType = KeyType;
    EXPECT_TRUE(checkOctreeInvariants(tree.data(), nNodes(tree)));

    int nParticles = mortonCodes.size();

    // check that referenced particles are within specified range
    for (std::size_t nodeIndex = 0; nodeIndex < nNodes(tree); ++nodeIndex)
    {
        int nodeStart = std::lower_bound(begin(mortonCodes), end(mortonCodes), tree[nodeIndex]) - begin(mortonCodes);

        int nodeEnd = std::lower_bound(begin(mortonCodes), end(mortonCodes), tree[nodeIndex + 1]) - begin(mortonCodes);

        // check that counts are correct
        EXPECT_EQ(nodeEnd - nodeStart, counts[nodeIndex]);
        if (!relaxBucketCount) { EXPECT_LE(counts[nodeIndex], bucketSize); }

        if (counts[nodeIndex]) { ASSERT_LT(nodeStart, nParticles); }

        for (std::size_t i = nodeStart; i < counts[nodeIndex]; ++i)
        {
            CodeType iCode = mortonCodes[i];
            EXPECT_TRUE(tree[nodeIndex] <= iCode);
            EXPECT_TRUE(iCode < tree[nodeIndex + 1]);
        }
    }
}

class ComputeOctreeTester : public testing::TestWithParam<int>
{
public:
    template<class KeyType>
    void check(int bucketSize)
    {
        int nParticles = 100000;

        std::vector<KeyType> keys = makeRandomGaussianKeys<KeyType>(nParticles);

        auto [tree, counts] = computeOctree<KeyType>(keys, bucketSize);

        checkOctreeWithCounts(tree, counts, bucketSize, keys, false);

        // update with unchanged particle codes
        updateOctree<KeyType>(keys, bucketSize, tree, counts);
        checkOctreeWithCounts(tree, counts, bucketSize, keys, false);

        // range of smallest treeNode
        KeyType minRange = std::numeric_limits<KeyType>::max();
        for (int i = 0; i < TreeNodeIndex(nNodes(tree)); ++i)
            minRange = std::min(minRange, tree[i + 1] - tree[i]);

        // change codes a bit
        std::mt19937 gen(42);
        std::uniform_int_distribution<std::make_signed_t<KeyType>> displace(-minRange, minRange);

        for (auto& key : keys)
            key = std::max(std::make_signed_t<KeyType>(0),
                           std::min(std::make_signed_t<KeyType>(key) + displace(gen),
                                    std::make_signed_t<KeyType>(nodeRange<KeyType>(0) - 1)));

        std::sort(begin(keys), end(keys));
        updateOctree<KeyType>(keys, bucketSize, tree, counts);
        // count < bucketSize may not be true anymore, but node counts still have to be correct
        checkOctreeWithCounts(tree, counts, bucketSize, keys, true);
    }
};

TEST_P(ComputeOctreeTester, pingPongRandomNormal32) { check<unsigned>(GetParam()); }

TEST_P(ComputeOctreeTester, pingPongRandomNormal64) { check<uint64_t>(GetParam()); }

std::array<int, 3> bucketSizesPP{64, 1024, 10000};

INSTANTIATE_TEST_SUITE_P(RandomBoxPP, ComputeOctreeTester, testing::ValuesIn(bucketSizesPP));

template<class KeyType>
void computeSpanningTree()
{
    {
        std::vector<KeyType> cornerstones{0, nodeRange<KeyType>(0)};
        std::vector<KeyType> spanningTree = computeSpanningTree<KeyType>(cornerstones);
        std::vector<KeyType> reference{0, nodeRange<KeyType>(0)};
        EXPECT_EQ(spanningTree, reference);
    }
    {
        std::vector<KeyType> cornerstones{0, pad(KeyType(1), 3), nodeRange<KeyType>(0)};
        std::vector<KeyType> spanningTree = computeSpanningTree<KeyType>(cornerstones);
        EXPECT_TRUE(checkOctreeInvariants(spanningTree.data(), nNodes(spanningTree)));
        EXPECT_EQ(spanningTree.size(), 9);
    }
    {
        std::vector<KeyType> cornerstones{0, 1, nodeRange<KeyType>(0) - 1, nodeRange<KeyType>(0)};
        std::vector<KeyType> spanningTree = computeSpanningTree<KeyType>(cornerstones);
        EXPECT_TRUE(checkOctreeInvariants(spanningTree.data(), nNodes(spanningTree)));
        if constexpr (std::is_same_v<KeyType, unsigned>)
            EXPECT_EQ(spanningTree.size(), 135);
        else
            EXPECT_EQ(spanningTree.size(), 289);
    }
}

TEST(CornerstoneOctree, computeSpanningTree)
{
    computeSpanningTree<unsigned>();
    computeSpanningTree<uint64_t>();
}

TEST(CornerstoneOctree, NodeDebug)
{
    using T       = double;
    using KeyType = uint64_t;
    KeyType a     = 0377400000000000000000lu;
    KeyType b     = 0500000000000000000000lu;
    std::vector<KeyType> sdomain{a, b};

    KeyType s0 = 0164640000000000000000lu;
    KeyType s1 = 0164650000000000000000lu;
    Box<T> box(0, 1, BoundaryType::periodic);
    IBox source                     = sfcIBox(sfcKey(s0), sfcKey(s1));
    auto [sourceCenter, sourceSize] = centerAndSize<KeyType>(source, box);
    unsigned prefixLength           = 3 * treeLevel(s1 - s0);
    KeyType sourcePrefix            = encodePlaceholderBit(s0, prefixLength);
    auto expCenter                  = sourceCenter;

    std::vector<KeyType> spanningTree(spanSfcRange(a, b) + 1);
    spanSfcRange(a, b, spanningTree.data());
    spanningTree.back() = b;

    std::vector<T> distances;
    T domainVol = 0;
    for (size_t i = 0; i < nNodes(spanningTree); ++i)
    {
        IBox target                     = sfcIBox(sfcKey(spanningTree[i]), sfcKey(spanningTree[i + 1]));
        auto [targetCenter, targetSize] = centerAndSize<KeyType>(target, box);

        auto distVec = minDistance(sourceCenter, sourceSize, targetCenter, targetSize, box);
        T distNorm   = std::sqrt(norm2(distVec));
        distances.push_back(distNorm);
        domainVol += 8 * targetSize[0] * targetSize[1] * targetSize[2];

        // T macSq = computeVecMacR2(sourcePrefix, expCenter, 1.0 / 0.5, box);
        T macSq = computeMinMacR2(sourcePrefix, invThetaMinMac(0.5), box)[3];

        bool isClose = evaluateMacPbc(expCenter, macSq, targetCenter, targetSize, box);
        std::cout << std::oct << spanningTree[i] << " - " << spanningTree[i + 1] << std::dec << ": " << distNorm
                  << " MAC " << (isClose ? "Fail" : "Pass") << std::endl;
    }
    T minSTDist = *std::min_element(distances.begin(), distances.end());
    std::cout << "Min source-target distance is " << minSTDist << std::endl;
    std::cout << "Domain volume is " << domainVol << ", cubeLength " << std::cbrt(domainVol) << std::endl;
}
