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
 * @brief Test source (mass) center computation
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/focus/source_center.hpp"
#include "cstone/tree/cs_util.hpp"

#include "coord_samples/random.hpp"

namespace cstone
{

TEST(FocusedOctree, sourceCenter)
{
    using T = double;

    {
        std::vector<T> x{-1, 2};
        std::vector<T> y{0, 0};
        std::vector<T> z{0, 0};
        std::vector<T> m{1, 1};

        SourceCenterType<T> probe = massCenter<T>(x.data(), y.data(), z.data(), m.data(), 0, 2);
        SourceCenterType<T> reference{0.5, 0, 0, 2};
        EXPECT_NEAR(probe[0], reference[0], 1e-10);
        EXPECT_NEAR(probe[1], reference[1], 1e-10);
        EXPECT_NEAR(probe[2], reference[2], 1e-10);
        EXPECT_NEAR(probe[3], reference[3], 1e-10);
    }
    {
        std::vector<T> x{0, 0, 0, 0, 1, 1, 1, 1};
        std::vector<T> y{0, 0, 1, 1, 0, 0, 1, 1};
        std::vector<T> z{0, 1, 0, 1, 0, 1, 0, 1};
        std::vector<T> m{2, 2, 2, 2, 2, 2, 2, 2};

        SourceCenterType<T> probe = massCenter<T>(x.data(), y.data(), z.data(), m.data(), 0, 8);
        SourceCenterType<T> reference{0.5, 0.5, 0.5, 16};
        EXPECT_NEAR(probe[0], reference[0], 1e-10);
        EXPECT_NEAR(probe[1], reference[1], 1e-10);
        EXPECT_NEAR(probe[2], reference[2], 1e-10);
        EXPECT_NEAR(probe[3], reference[3], 1e-10);
    }
}

template<class KeyType>
static void computeSourceCenter()
{
    using T                 = double;
    LocalIndex numParticles = 20000;
    Box<T> box{-1, 1};
    unsigned csBucketSize = 16;

    RandomGaussianCoordinates<T, SfcKind<KeyType>> coords(numParticles, box);

    auto [csTree, csCounts] =
        computeOctree(coords.particleKeys().data(), coords.particleKeys().data() + numParticles, csBucketSize);
    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(csTree));
    updateInternalTree<KeyType>(csTree, octree.data());

    std::vector<T> masses(numParticles);
    std::generate(begin(masses), end(masses), [numParticles]() { return drand48() / numParticles; });
    std::vector<util::array<T, 4>> centers(octree.numNodes);

    std::vector<LocalIndex> layout(octree.numLeafNodes + 1, 0);
    std::inclusive_scan(csCounts.begin(), csCounts.end(), layout.begin() + 1);

    auto toInternal = leafToInternal(octree);
    computeLeafMassCenter<T, T, T>(coords.x(), coords.y(), coords.z(), masses, toInternal, layout.data(),
                                   centers.data());
    upsweep(octree.levelRange, octree.childOffsets, centers.data(), CombineSourceCenter<T>{});

    util::array<T, 4> refRootCenter =
        massCenter<T>(coords.x().data(), coords.y().data(), coords.z().data(), masses.data(), 0, numParticles);

    TreeNodeIndex rootNode = octree.levelRange[0];

    EXPECT_NEAR(centers[rootNode][3], refRootCenter[3], 1e-8);
}

TEST(FocusedOctree, sourceCenterUpsweep)
{
    computeSourceCenter<unsigned>();
    computeSourceCenter<uint64_t>();
}

} // namespace cstone