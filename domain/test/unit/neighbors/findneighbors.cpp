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
 * @brief Neighbor search tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "cstone/findneighbors.hpp"
#include "coord_samples/random.hpp"

#include "all_to_all.hpp"

using namespace cstone;

TEST(FindNeighbors, distanceSqPbc)
{
    {
        Box<double> box(0, 10, BoundaryType::open);
        EXPECT_DOUBLE_EQ(64.0, distanceSq<true>(1., 0., 0., 9., 0., 0., box));
        EXPECT_DOUBLE_EQ(64.0, distanceSq<true>(9., 0., 0., 1., 0., 0., box));
        EXPECT_DOUBLE_EQ(192.0, distanceSq<true>(9., 9., 9., 1., 1., 1., box));
    }
    {
        Box<double> box(0, 10, BoundaryType::periodic);
        EXPECT_DOUBLE_EQ(4.0, distanceSq<true>(1., 0., 0., 9., 0., 0., box));
        EXPECT_DOUBLE_EQ(4.0, distanceSq<true>(9., 0., 0., 1., 0., 0., box));
        EXPECT_DOUBLE_EQ(12.0, distanceSq<true>(9., 9., 9., 1., 1., 1., box));
    }
}

template<class Coordinates, class T>
void neighborCheck(const Coordinates& coords, T radius, const Box<T>& box)
{
    using KeyType        = typename Coordinates::KeyType::ValueType;
    cstone::LocalIndex n = coords.x().size();
    unsigned ngmax       = n;

    std::vector<T> h(n, radius / 2);

    std::vector<LocalIndex> neighborsRef(n * ngmax);
    std::vector<unsigned> neighborsCountRef(n);
    all2allNeighbors(coords.x().data(), coords.y().data(), coords.z().data(), h.data(), n, neighborsRef.data(),
                     neighborsCountRef.data(), ngmax, box);
    sortNeighbors(neighborsRef.data(), neighborsCountRef.data(), n, ngmax);

    std::vector<LocalIndex> neighborsProbe(n * ngmax);
    std::vector<unsigned> neighborsCountProbe(n);

    auto sfcKeys = coords.particleKeys().data();

    unsigned bucketSize   = 64;
    auto [csTree, counts] = computeOctree(sfcKeys, sfcKeys + n, bucketSize);
    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(csTree));
    updateInternalTree<KeyType>(csTree, octree.data());

    std::vector<LocalIndex> layout(nNodes(csTree) + 1, 0);
    std::inclusive_scan(counts.begin(), counts.end(), layout.begin() + 1);

    gsl::span<const KeyType> nodeKeys(octree.prefixes.data(), octree.numNodes);
    std::vector<Vec3<T>> centers(octree.numNodes), sizes(octree.numNodes);
    nodeFpCenters<KeyType>(nodeKeys, centers.data(), sizes.data(), box);

    OctreeNsView<T, KeyType> nsView{octree.numLeafNodes,
                                    octree.prefixes.data(),
                                    octree.childOffsets.data(),
                                    octree.internalToLeaf.data(),
                                    octree.levelRange.data(),
                                    nullptr,
                                    layout.data(),
                                    centers.data(),
                                    sizes.data()};

    findNeighbors(coords.x().data(), coords.y().data(), coords.z().data(), h.data(), 0, n, box, nsView, ngmax,
                  neighborsProbe.data(), neighborsCountProbe.data());
    sortNeighbors(neighborsProbe.data(), neighborsCountProbe.data(), n, ngmax);

    EXPECT_EQ(neighborsRef, neighborsProbe);
    EXPECT_EQ(neighborsCountRef, neighborsCountProbe);
}

class FindNeighborsRandom
    : public testing::TestWithParam<std::tuple<double, int, std::array<double, 6>, cstone::BoundaryType>>
{
public:
    template<class KeyType, template<class...> class CoordinateKind>
    void check()
    {
        double radius                = std::get<0>(GetParam());
        int nParticles               = std::get<1>(GetParam());
        std::array<double, 6> limits = std::get<2>(GetParam());
        cstone::BoundaryType usePbc  = std::get<3>(GetParam());
        Box<double> box{limits[0], limits[1], limits[2], limits[3], limits[4], limits[5], usePbc, usePbc, usePbc};

        CoordinateKind<double, KeyType> coords(nParticles, box);

        neighborCheck(coords, radius, box);
    }
};

TEST_P(FindNeighborsRandom, HilbertUniform32) { check<HilbertKey<uint32_t>, RandomCoordinates>(); }
TEST_P(FindNeighborsRandom, HilbertUniform64) { check<HilbertKey<uint64_t>, RandomCoordinates>(); }
TEST_P(FindNeighborsRandom, HilbertGaussian32) { check<HilbertKey<uint32_t>, RandomGaussianCoordinates>(); }
TEST_P(FindNeighborsRandom, HilbertGaussian64) { check<HilbertKey<uint64_t>, RandomGaussianCoordinates>(); }

std::array<double, 2> radii{0.124, 0.0624};
std::array<int, 1> nParticles{2500};
std::array<std::array<double, 6>, 2> boxes{{{0., 1., 0., 1., 0., 1.}, {-1.2, 0.23, -0.213, 3.213, -5.1, 1.23}}};
std::array<cstone::BoundaryType, 2> pbcUsage{BoundaryType::open, BoundaryType::periodic};

INSTANTIATE_TEST_SUITE_P(RandomNeighbors,
                         FindNeighborsRandom,
                         testing::Combine(testing::ValuesIn(radii),
                                          testing::ValuesIn(nParticles),
                                          testing::ValuesIn(boxes),
                                          testing::ValuesIn(pbcUsage)));

INSTANTIATE_TEST_SUITE_P(RandomNeighborsLargeRadius,
                         FindNeighborsRandom,
                         testing::Combine(testing::Values(3.0),
                                          testing::Values(500),
                                          testing::ValuesIn(boxes),
                                          testing::ValuesIn(pbcUsage)));
