/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
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
void neighborCheck(const Coordinates& coords, T radius, const Box<T>& box, const bool disableMixD = true)
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

    unsigned bucketSize   = 64;
    auto [csTree, counts] = computeOctree<KeyType>(coords.particleKeys(), bucketSize);
    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(csTree));
    updateInternalTree<KeyType>(csTree, octree.data());

    std::vector<LocalIndex> layout(nNodes(csTree) + 1, 0);
    std::inclusive_scan(counts.begin(), counts.end(), layout.begin() + 1);

    std::span<const KeyType> nodeKeys(octree.prefixes.data(), octree.numNodes);
    std::vector<Vec3<T>> centers(octree.numNodes), sizes(octree.numNodes);
    nodeFpCenters<KeyType>(nodeKeys, centers.data(), sizes.data(), box, disableMixD);

    OctreeNsView<T, KeyType> nsView{octree.numLeafNodes,
                                    octree.prefixes.data(),
                                    octree.childOffsets.data(),
                                    octree.parents.data(),
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
    template<class KeyType, template<class...> class CoordinateKind, bool disableMixD = true>
    void check()
    {
        double radius                = std::get<0>(GetParam());
        int nParticles               = std::get<1>(GetParam());
        std::array<double, 6> limits = std::get<2>(GetParam());
        cstone::BoundaryType usePbc  = std::get<3>(GetParam());
        Box<double> box{limits[0], limits[1], limits[2], limits[3], limits[4], limits[5], usePbc, usePbc, usePbc};

        const auto mixDBits = getBoxMixDimensionBits<double, KeyType, Box<double>>(box);
        auto coords         = disableMixD
                                  ? CoordinateKind<double, KeyType>(nParticles, box)
                                  : CoordinateKind<double, KeyType>(nParticles, box, 42, mixDBits.bx, mixDBits.by, mixDBits.bz);

        neighborCheck(coords, radius, box, disableMixD);
    }
};

class CompareNeighborsRandomMixD
    : public testing::TestWithParam<std::tuple<int, std::array<double, 6>, cstone::BoundaryType>>
{
public:
    template<class KeyType3D, class KeyTypeMixD, template<class...> class CoordinateKind>
    void check()
    {
        int nParticles              = std::get<0>(GetParam());
        auto limits                 = std::get<1>(GetParam());
        cstone::BoundaryType usePbc = std::get<2>(GetParam());
        Box<double> box{limits[0], limits[1], limits[2], limits[3], limits[4], limits[5], usePbc, usePbc, usePbc};

        const auto mixDBits = getBoxMixDimensionBits<double, KeyTypeMixD, Box<double>>(box);

        const CoordinateKind<double, KeyType3D> coords3D(nParticles, box, 42);
        const CoordinateKind<double, KeyTypeMixD> coordsMixD(nParticles, box, 42, mixDBits.bx, mixDBits.by,
                                                             mixDBits.bz);

        unsigned bucketSize       = 64;
        auto [csTree3D, counts3D] = computeOctree<typename KeyType3D::ValueType>(coords3D.particleKeys(), bucketSize);
        auto [csTreeMixD, countsMixD] =
            computeOctree<typename KeyTypeMixD::ValueType>(coordsMixD.particleKeys(), bucketSize);

        int leafCount3D = 0;
        for (size_t i = 0; i < csTree3D.size(); ++i)
        {
            if (counts3D[i] > 0) { leafCount3D++; }
        }

        int leafCountMixD = 0;
        for (size_t i = 0; i < csTreeMixD.size(); ++i)
        {
            if (countsMixD[i] > 0) { leafCountMixD++; }
        }
        EXPECT_LE(leafCountMixD, leafCount3D);
    }
};

TEST_P(FindNeighborsRandom, HilbertUniform32) { check<HilbertKey<uint32_t>, RandomCoordinates>(); }
TEST_P(FindNeighborsRandom, HilbertUniform64) { check<HilbertKey<uint64_t>, RandomCoordinates>(); }
TEST_P(FindNeighborsRandom, HilbertMixDUniform32) { check<HilbertMixDKey<uint32_t>, RandomCoordinates, false>(); }
TEST_P(FindNeighborsRandom, HilbertMixDUniform64) { check<HilbertMixDKey<uint64_t>, RandomCoordinates, false>(); }
TEST_P(FindNeighborsRandom, HilbertGaussian32) { check<HilbertKey<uint32_t>, RandomGaussianCoordinates>(); }
TEST_P(FindNeighborsRandom, HilbertGaussian64) { check<HilbertKey<uint64_t>, RandomGaussianCoordinates>(); }

TEST_P(CompareNeighborsRandomMixD, Hilbert3DvsMixDUniform32)
{
    check<HilbertKey<uint32_t>, HilbertMixDKey<uint32_t>, RandomCoordinates>();
}
TEST_P(CompareNeighborsRandomMixD, Hilbert3DvsMixDUniform64)
{
    check<HilbertKey<uint64_t>, HilbertMixDKey<uint64_t>, RandomCoordinates>();
}
TEST_P(CompareNeighborsRandomMixD, Hilbert3DvsMixDGaussian32)
{
    check<HilbertKey<uint32_t>, HilbertMixDKey<uint32_t>, RandomGaussianCoordinates>();
}
TEST_P(CompareNeighborsRandomMixD, Hilbert3DvsMixDGaussian64)
{
    check<HilbertKey<uint64_t>, HilbertMixDKey<uint64_t>, RandomGaussianCoordinates>();
}

std::array<double, 2> radii{0.124, 0.0624};
std::array<int, 1> nParticles{2500};
std::array<std::array<double, 6>, 2> boxes{{{0., 1., 0., 1., 0., 1.}, {-1.2, 0.23, -0.213, 3.213, -5.1, 1.23}}};
std::array<double, 3> radiiMixD{0.001, 1., 0.01};
std::array<std::array<double, 6>, 2> boxesMixD{
    {{0., 1., 0., 0.015625, 0., 0.00390625}, {0., 0.00390625, 0., 1., 0., 0.015625}}};
std::array<cstone::BoundaryType, 2> pbcUsage{BoundaryType::open, BoundaryType::periodic};

INSTANTIATE_TEST_SUITE_P(RandomNeighbors,
                         FindNeighborsRandom,
                         testing::Combine(testing::ValuesIn(radii),
                                          testing::ValuesIn(nParticles),
                                          testing::ValuesIn(boxes),
                                          testing::ValuesIn(pbcUsage)));

INSTANTIATE_TEST_SUITE_P(RandomNeighborsMixD,
                         FindNeighborsRandom,
                         testing::Combine(testing::ValuesIn(radiiMixD),
                                          testing::ValuesIn(nParticles),
                                          testing::ValuesIn(boxesMixD),
                                          testing::ValuesIn(pbcUsage)));

INSTANTIATE_TEST_SUITE_P(CompareRandomNeighbors3DMixD,
                         CompareNeighborsRandomMixD,
                         testing::Combine(testing::ValuesIn(nParticles),
                                          testing::ValuesIn(boxesMixD),
                                          testing::ValuesIn(pbcUsage)));

INSTANTIATE_TEST_SUITE_P(RandomNeighborsLargeRadius,
                         FindNeighborsRandom,
                         testing::Combine(testing::Values(3.0),
                                          testing::Values(500),
                                          testing::ValuesIn(boxes),
                                          testing::ValuesIn(pbcUsage)));
