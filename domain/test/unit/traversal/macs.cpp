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

TEST(Macs, minMacMutual)
{
    using T = double;

    Vec3<T> cA{0.5, 0.5, 0.5};
    Vec3<T> sA{0.5, 0.5, 0.5};

    Vec3<T> cB{3.5, 3.5, 3.5};
    Vec3<T> sB{0.5, 0.5, 0.5};

    EXPECT_TRUE(minMacMutual(cA, sA, cB, sB, Box<T>(0, 4, BoundaryType::open), 1.0 / 0.29));
    EXPECT_FALSE(minMacMutual(cA, sA, cB, sB, Box<T>(0, 4, BoundaryType::open), 1.0 / 0.28));

    EXPECT_FALSE(minMacMutual(cA, sA, cB, sB, Box<T>(0, 4, BoundaryType::periodic), 1.0));
}

TEST(Macs, minVecMacMutual)
{
    using T = double;

    Vec3<T> cA{0.5, 0.5, 0.5};
    Vec3<T> sA{0.5, 0.5, 0.5};

    Vec3<T> cB{3.5, 3.5, 3.5};
    Vec3<T> sB{0.5, 0.5, 0.5};

    EXPECT_TRUE(minVecMacMutual(cA, sA, cB, sB, Box<T>(0, 4, BoundaryType::open), invThetaVecMac(0.39)));
    EXPECT_FALSE(minVecMacMutual(cA, sA, cB, sB, Box<T>(0, 4, BoundaryType::open), invThetaVecMac(0.38)));

    EXPECT_FALSE(minVecMacMutual(cA, sA, cB, sB, Box<T>(0, 4, BoundaryType::periodic), invThetaVecMac(1.0)));
}

template<class KeyType, class T>
static std::vector<char> markVecMacAll2All(const KeyType* leaves,
                                           gsl::span<const KeyType> prefixes,
                                           const Vec4<T>* centers,
                                           TreeNodeIndex firstLeaf,
                                           TreeNodeIndex lastLeaf,
                                           const Box<T>& box)
{
    std::vector<char> markings(prefixes.size(), 0);

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

    auto [leaves, counts] = computeOctree(coords.particleKeys().data(),
                                          coords.particleKeys().data() + coords.particleKeys().size(), bucketSize);
    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(leaves));
    updateInternalTree<KeyType>(leaves, octree.data());

    std::vector<LocalIndex> layout(octree.numLeafNodes + 1, 0);
    std::inclusive_scan(counts.begin(), counts.end(), layout.begin() + 1);

    auto toInternal = leafToInternal(octree);

    std::vector<SourceCenterType<T>> centers(octree.numNodes);
    computeLeafMassCenter<T, T, T>(coords.x(), coords.y(), coords.z(), masses, toInternal, layout.data(),
                                   centers.data());
    upsweep(octree.levelRange, octree.childOffsets, centers.data(), CombineSourceCenter<T>{});
    setMac<T, KeyType>(octree.prefixes, centers, 1.0 / theta, box);

    std::vector<char> markings(octree.numNodes, 0);

    TreeNodeIndex focusIdxStart = 4;
    TreeNodeIndex focusIdxEnd   = 22;

    markMacs(octree.prefixes.data(), octree.childOffsets.data(), centers.data(), box, leaves.data() + focusIdxStart,
             focusIdxEnd - focusIdxStart, false, markings.data());

    std::vector<char> reference =
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

    std::vector<char> macs(ov.numNodes, 0);
    markMacs(ov.prefixes, ov.childOffsets, centers.data(), box, leaves.data() + 0, 32, true, macs.data());

    std::vector<char> macRef{1, 0, 0, 0, 0, 1, 1, 1, 1};
    macRef.resize(ov.numNodes);
    EXPECT_EQ(macRef, macs);

    std::fill(macs.begin(), macs.end(), 0);
    markMacs(ov.prefixes, ov.childOffsets, centers.data(), box, leaves.data() + 0, 32, false, macs.data());
    int numMacs = std::accumulate(macs.begin(), macs.end(), 0);
    EXPECT_EQ(numMacs, 5 + 16);
}

} // namespace cstone
