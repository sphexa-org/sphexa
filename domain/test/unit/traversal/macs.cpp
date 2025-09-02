/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Test multipole acceptance criteria
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/focus/source_center.hpp"
#include "cstone/traversal/macs.hpp"
#include "cstone/tree/cs_util.hpp"

#include "coord_samples/random.hpp"

namespace cstone
{

TEST(Macs, evaluateMAC)
{
    using T = double;

    Box<T> noPbcBox(0, 1, BoundaryType::open);
    Box<T> box(0, 1, BoundaryType::periodic);

    Vec3<T> tcenter{0.1, 0.1, 0.1};
    Vec3<T> tsize{0.01, 0.01, 0.01};

    // R = sqrt(0.03) = 0.173
    T mac = 0.03;
    {
        Vec3<T> scenter{0.2, 0.2, 0.2};
        EXPECT_TRUE(evaluateMacPbc(scenter, mac, tcenter, tsize, noPbcBox));
        EXPECT_TRUE(evaluateMacPbc(scenter, mac, tcenter, tsize, box));
    }
    {
        Vec3<T> scenter{0.2101, 0.2101, 0.2101};
        EXPECT_FALSE(evaluateMacPbc(scenter, mac, tcenter, tsize, noPbcBox));
        EXPECT_FALSE(evaluateMacPbc(scenter, mac, tcenter, tsize, box));
    }
    {
        Vec3<T> scenter{1.0, 1.0, 1.0};
        EXPECT_TRUE(evaluateMacPbc(scenter, mac, tcenter, tsize, box));
    }
    {
        Vec3<T> scenter{0.9899, 0.9899, 0.9899};
        EXPECT_FALSE(evaluateMacPbc(scenter, mac, tcenter, tsize, box));
    }
}

TEST(Macs, minMacMutualInt)
{
    using KeyType = uint32_t;
    using T       = double;

    int maxCoord = 1u << maxTreeLevel<KeyType>{};

    IBox b(100, 110, 100, 110, 100, 105);
    float invTheta = 1.5;
    {
        Box<T> box(0, 1);
        auto ellipse = Vec3<T>{box.ilx(), box.ily(), box.ilz()} * box.maxExtent() * invTheta;

        EXPECT_TRUE(minMacMutualInt({83, 84, 100, 101, 100, 101}, b, ellipse, {0, 0, 0}));
        EXPECT_FALSE(minMacMutualInt({84, 85, 100, 101, 100, 101}, b, ellipse, {0, 0, 0}));
        EXPECT_TRUE(minMacMutualInt({100, 101, 83, 84, 100, 101}, b, ellipse, {0, 0, 0}));
        EXPECT_FALSE(minMacMutualInt({100, 101, 84, 85, 100, 101}, b, ellipse, {0, 0, 0}));
        EXPECT_TRUE(minMacMutualInt({100, 101, 100, 101, 121, 122}, b, ellipse, {0, 0, 0}));
        EXPECT_FALSE(minMacMutualInt({100, 101, 100, 101, 120, 121}, b, ellipse, {0, 0, 0}));

        EXPECT_TRUE(minMacMutualInt({90, 91}, b, ellipse, {0, 0, 0}));
        EXPECT_FALSE(minMacMutualInt({91, 92}, b, ellipse, {0, 0, 0}));
    }
    {
        Box<T> box(0, 2, 0, 1, 0, 2);
        auto ellipse = Vec3<T>{box.ilx(), box.ily(), box.ilz()} * box.maxExtent() * invTheta;
        EXPECT_TRUE(minMacMutualInt({83, 84, 100, 101, 100, 101}, b, ellipse, {0, 0, 0}));
        EXPECT_FALSE(minMacMutualInt({84, 85, 100, 101, 100, 101}, b, ellipse, {0, 0, 0}));
        EXPECT_TRUE(minMacMutualInt({100, 101, 68, 69, 100, 101}, b, ellipse, {0, 0, 0}));
        EXPECT_FALSE(minMacMutualInt({100, 101, 69, 70, 100, 101}, b, ellipse, {0, 0, 0}));
    }
    {
        auto pbc_t = BoundaryType::periodic;
        Box<T> box(0, 1, pbc_t);
        auto pbc = Vec3<int>{box.boundaryX() == pbc_t, box.boundaryY() == pbc_t, box.boundaryZ() == pbc_t} * maxCoord;
        auto ellipse = Vec3<T>{box.ilx(), box.ily(), box.ilz()} * box.maxExtent() * invTheta;
        EXPECT_TRUE(minMacMutualInt({1023 - 67, 1023, 100, 101, 100, 101}, b, ellipse, pbc));
        EXPECT_FALSE(minMacMutualInt({1023 - 68, 1023, 100, 101, 100, 101}, b, ellipse, pbc));
    }
}

template<class KeyType, class T>
static std::vector<uint8_t> markVecMacAll2All(const KeyType* leaves,
                                              std::span<const KeyType> prefixes,
                                              const Vec4<T>* centers,
                                              TreeNodeIndex firstLeaf,
                                              TreeNodeIndex lastLeaf,
                                              const Box<T>& box)
{
    std::vector<uint8_t> markings(prefixes.size(), 0);

    // loop over target cells
    for (TreeNodeIndex i = firstLeaf; i < lastLeaf; ++i)
    {
        IBox targetBox                  = sfcIBox(sfcKey(leaves[i]), sfcKey(leaves[i + 1]));
        auto [targetCenter, targetSize] = centerAndSize<KeyType>(targetBox, box);

        // loop over source cells
        for (size_t j = 0; j < prefixes.size(); ++j)
        {
            // source cells must not be in target cell range
            KeyType jstart = decodePlaceholderBit(prefixes[j]);
            KeyType jend   = jstart + nodeRange<KeyType>(decodePrefixLength(prefixes[j]) / 3);
            if (leaves[firstLeaf] <= jstart && jend <= leaves[lastLeaf]) { continue; }

            Vec4<T> center   = centers[j];
            bool violatesMac = evaluateMacPbc(makeVec3(center), center[3], targetCenter, targetSize, box);
            if (violatesMac) { markings[j] = 1; }
        }
    }

    return markings;
}

template<class KeyType>
static void markMacVector()
{
    using T                 = double;
    LocalIndex numParticles = 1000;
    unsigned bucketSize     = 2;
    float theta             = 0.58;
    Box<T> box(0, 1);

    RandomGaussianCoordinates<T, SfcKind<KeyType>> coords(numParticles, box);
    std::vector<T> masses(numParticles, 1.0 / numParticles);

    auto [leaves, counts] = computeOctree<KeyType>(coords.particleKeys(), bucketSize);
    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(leaves));
    updateInternalTree<KeyType>(leaves, octree.data());

    std::vector<LocalIndex> layout(octree.numLeafNodes + 1, 0);
    std::inclusive_scan(counts.begin(), counts.end(), layout.begin() + 1);

    auto toInternal = leafToInternal(octree);

    std::vector<SourceCenterType<T>> centers(octree.numNodes);
    computeLeafMassCenter<T, T, T>(coords.x(), coords.y(), coords.z(), masses, toInternal, layout.data(),
                                   centers.data());
    upsweep(octree.levelRange, octree.childOffsets.data(), centers.data(), CombineSourceCenter<T>{});
    setMac<T, KeyType>(octree.prefixes, centers, 1.0 / theta, box);

    std::vector<uint8_t> markings(octree.numNodes, 0);

    TreeNodeIndex focusIdxStart = 4;
    TreeNodeIndex focusIdxEnd   = 22;

    markMacs(octree.prefixes.data(), octree.childOffsets.data(), octree.parents.data(), centers.data(), box,
             leaves.data() + focusIdxStart, focusIdxEnd - focusIdxStart, false, markings.data());

    std::vector<uint8_t> reference =
        markVecMacAll2All<KeyType>(leaves.data(), octree.prefixes, centers.data(), focusIdxStart, focusIdxEnd, box);

    EXPECT_EQ(markings, reference);
}

TEST(Macs, markMacVector)
{
    markMacVector<unsigned>();
    markMacVector<uint64_t>();
}

TEST(Macs, limitSource4x4)
{
    using KeyType = uint64_t;
    using T       = double;

    Box<T> box(0, 1);
    float invTheta = sqrt(3.) / 2;

    std::vector<KeyType> leaves = makeUniformNLevelTree<KeyType>(64, 1);
    OctreeData<KeyType, CpuTag> fullTree;
    fullTree.resize(nNodes(leaves));
    OctreeView<KeyType> ov = fullTree.data();
    updateInternalTree<KeyType>(leaves, ov);

    std::vector<SourceCenterType<T>> centers(ov.numNodes);
    geoMacSpheres<KeyType>({ov.prefixes, size_t(ov.numNodes)}, centers.data(), invTheta, box);

    std::vector<uint8_t> macs(ov.numNodes, 0);
    markMacs(ov.prefixes, ov.childOffsets, ov.parents, centers.data(), box, leaves.data() + 0, 32, true, macs.data());

    std::vector<uint8_t> macRef{1, 0, 0, 0, 0, 1, 1, 1, 1};
    macRef.resize(ov.numNodes);
    EXPECT_EQ(macRef, macs);

    std::fill(macs.begin(), macs.end(), 0);
    markMacs(ov.prefixes, ov.childOffsets, ov.parents, centers.data(), box, leaves.data() + 0, 32, false, macs.data());
    int numMacs = std::accumulate(macs.begin(), macs.end(), 0);
    EXPECT_EQ(numMacs, 5 + 16);
}

} // namespace cstone
