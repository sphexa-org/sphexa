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

    unsigned bucketSize   = 64;
    auto [csTree, counts] = computeOctree<KeyType>(coords.particleKeys(), bucketSize);
    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(csTree));
    updateInternalTree<KeyType>(csTree, octree.data());

    std::vector<LocalIndex> layout(nNodes(csTree) + 1, 0);
    std::inclusive_scan(counts.begin(), counts.end(), layout.begin() + 1);

    std::span<const KeyType> nodeKeys(octree.prefixes.data(), octree.numNodes);
    std::vector<Vec3<T>> centers(octree.numNodes), sizes(octree.numNodes);
    nodeFpCenters<KeyType>(nodeKeys, centers.data(), sizes.data(), box, true);

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
    std::cout << "octree.numLeafNodes = " << octree.numLeafNodes << std::endl;
    std::cout << "octree.prefixes.size() = " << octree.prefixes.size() << std::endl;
    std::cout << "octree.childOffsets.size() = " << octree.childOffsets.size() << std::endl;
    std::cout << "octree.internalToLeaf.size() = " << octree.internalToLeaf.size() << std::endl;
    std::cout << "octree.levelRange.size() = " << octree.levelRange.size() << std::endl;
    std::cout << "layout.size() = " << layout.size() << std::endl;
    std::cout << "centers.size() = " << centers.size() << std::endl;
    std::cout << "sizes.size() = " << sizes.size() << std::endl;
    // for (size_t i{}; i < octree.prefixes.size(); ++i)
    // {
    //     std::cout << "octree.prefixes[" << i << "] = " << std::oct << octree.prefixes[i] << std::dec << " center: ("
    //               << centers[i][0] << ", " << centers[i][1] << ", " << centers[i][2] << ") size: (" << sizes[i][0]
    //               << ", " << sizes[i][1] << ", " << sizes[i][2] << ")" << std::endl;
    // }
    findNeighbors(coords.x().data(), coords.y().data(), coords.z().data(), h.data(), 0, n, box, nsView, ngmax,
                  neighborsProbe.data(), neighborsCountProbe.data());
    sortNeighbors(neighborsProbe.data(), neighborsCountProbe.data(), n, ngmax);
    std::cout << "neighborsRef.size() = " << neighborsRef.size() << std::endl;
    std::cout << "neighborsProbe.size() = " << neighborsProbe.size() << std::endl;
    std::cout << "neighborsCountRef.size() = " << neighborsCountRef.size() << std::endl;
    std::cout << "neighborsCountProbe.size() = " << neighborsCountProbe.size() << std::endl;
    EXPECT_EQ(neighborsRef, neighborsProbe);
    EXPECT_EQ(neighborsCountRef, neighborsCountProbe);
}

template<class Coordinates, class T>
void neighborCheckMixD(const Coordinates& coords, T radius, const Box<T>& box, unsigned bx, unsigned by, unsigned bz)
{
    using KeyType        = typename Coordinates::KeyType::ValueType;
    cstone::LocalIndex n = coords.x().size();
    unsigned ngmax       = n;

    std::vector<T> h(n, radius / 2);

    std::vector<LocalIndex> neighborsRef(n * ngmax);
    std::vector<unsigned> neighborsCountRef(n);
    // std::cout << "Particle coordinates:\n";
    // for (size_t i = 0; i < n; ++i)
    // {
    //     std::cout << "Particle " << i << ": (" << coords.x()[i] << ", " << coords.y()[i] << ", " << coords.z()[i] << ")\n";
    // }
    all2allNeighbors(coords.x().data(), coords.y().data(), coords.z().data(), h.data(), n, neighborsRef.data(),
                     neighborsCountRef.data(), ngmax, box);
    sortNeighbors(neighborsRef.data(), neighborsCountRef.data(), n, ngmax);

    std::vector<LocalIndex> neighborsProbe(n * ngmax);
    std::vector<unsigned> neighborsCountProbe(n);

    unsigned bucketSize   = 64;
    auto [csTree, counts] = computeOctree<KeyType>(coords.particleKeys(), bucketSize);
    for (size_t i{}; i < csTree.size(); ++i)
    {
        std::cout << "csTree[" << i << "] = " << std::oct << csTree[i] << std::dec << " count: " << counts[i]
                  << std::endl;
    }
    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(csTree));
    updateInternalTree<KeyType>(csTree, octree.data());
    // EXPECT_EQ(encodePlaceholderBit(csTree[0], 6), octree.prefixes[9]);
    // EXPECT_EQ(encodePlaceholderBit(csTree[1], 6), octree.prefixes[10]);
    // EXPECT_EQ(encodePlaceholderBit(csTree[8], 6), octree.prefixes[17]);
    // EXPECT_EQ(encodePlaceholderBit(csTree[9], 6), octree.prefixes[18]);
    std::vector<LocalIndex> layout(nNodes(csTree) + 1);
    std::exclusive_scan(counts.begin(), counts.end() + 1, layout.begin(), 0);

    std::span<const KeyType> nodeKeys(octree.prefixes.data(), octree.numNodes);
    int number_of_non_zero_leaves{};
    std::cout << "nodeKeysMixD\n";
    for (size_t i = 0; i < csTree.size(); ++i)
    {
        if (counts[i] > 0) { number_of_non_zero_leaves++; }
        std::cout << i << ": " << std::oct << csTree[i] << std::dec << " count: " << counts[i] << std::endl;
    }
    std::cout << "number_of_non_zero_leaves = " << number_of_non_zero_leaves << std::endl;
    std::vector<Vec3<T>> centers(octree.numNodes), sizes(octree.numNodes);
    nodeFpCenters<KeyType>(nodeKeys, centers.data(), sizes.data(), box, bx, by, bz);

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
    std::set<std::pair<Vec3<T>, Vec3<T>>> uniqueBoxes;
    for (size_t i = 0; i < octree.numNodes; ++i)
    {
        if (sizes[i][0] != 0 || sizes[i][1] != 0 || sizes[i][2] != 0)
        {
            auto boxPair = std::make_pair(centers[i], sizes[i]);
            if (uniqueBoxes.find(boxPair) != uniqueBoxes.end())
            {
                ADD_FAILURE() << "Duplicate box found: center = (" << centers[i][0] << ", " << centers[i][1] << ", "
                              << centers[i][2] << "), size = (" << sizes[i][0] << ", " << sizes[i][1] << ", "
                              << sizes[i][2] << ")";
            }
            uniqueBoxes.insert(boxPair);
        }
    }
    std::cout << "octree.numLeafNodes = " << octree.numLeafNodes << std::endl;
    std::cout << "octree.prefixes.size() = " << octree.prefixes.size() << std::endl;
    std::cout << "octree.childOffsets.size() = " << octree.childOffsets.size() << std::endl;
    std::cout << "octree.internalToLeaf.size() = " << octree.internalToLeaf.size() << std::endl;
    std::cout << "octree.levelRange.size() = " << octree.levelRange.size() << std::endl;
    std::cout << "layout.size() = " << layout.size() << std::endl;
    std::cout << "centers.size() = " << centers.size() << std::endl;
    std::cout << "sizes.size() = " << sizes.size() << std::endl;
    for (size_t i{}; i < octree.prefixes.size(); ++i)
    {
        std::cout << "octree.prefixes[" << i << "] = " << std::oct << octree.prefixes[i] << std::dec << " center: ("
                  << centers[i][0] << ", " << centers[i][1] << ", " << centers[i][2] << ") size: (" << sizes[i][0]
                  << ", " << sizes[i][1] << ", " << sizes[i][2] << ")" << std::endl;
    }
    // for (size_t i{}; i < layout.size(); ++i)
    // {
    //     std::cout << "layout[" << i << "] = " << layout[i] << std::endl;
    // }

    findNeighbors(coords.x().data(), coords.y().data(), coords.z().data(), h.data(), 0, n, box, nsView, ngmax,
                  neighborsProbe.data(), neighborsCountProbe.data());
    sortNeighbors(neighborsProbe.data(), neighborsCountProbe.data(), n, ngmax);

    std::cout << "neighborsRef.size() = " << neighborsRef.size() << std::endl;
    std::cout << "neighborsProbe.size() = " << neighborsProbe.size() << std::endl;
    std::cout << "neighborsCountRef.size() = " << neighborsCountRef.size() << std::endl;
    std::cout << "neighborsCountProbe.size() = " << neighborsCountProbe.size() << std::endl;
    for (size_t i = 0; i < neighborsRef.size(); ++i)
    {
        if (neighborsRef[i] != neighborsProbe[i])
        {
            std::cout << "neighborsRef[" << i / n << "][" << i % n << "]: " << neighborsRef[i] << " neighborsProbe["
                      << i / n << "][" << i % n << "]: " << neighborsProbe[i] << std::endl;
            std::cout << "particle a: (" << coords.x()[i % n] << ", " << coords.y()[i % n] << ", " << coords.z()[i % n]
                      << ") ";
            std::cout << "particle b: (" << coords.x()[neighborsRef[i]] << ", " << coords.y()[neighborsRef[i]] << ", "
                      << coords.z()[neighborsRef[i]] << ") ";
            std::cout << "distance: "
                      << std::sqrt(distanceSq<false>(coords.x()[i % n], coords.y()[i % n], coords.z()[i % n],
                                                     coords.x()[neighborsRef[i]], coords.y()[neighborsRef[i]],
                                                     coords.z()[neighborsRef[i]], box))
                      << std::endl;
        }
    }
    for (size_t i = 0; i < neighborsCountRef.size(); ++i)
    {
        if (neighborsCountRef[i] != neighborsCountProbe[i])
        {
            std::cout << "neighborsCountRef[" << i << "]: " << neighborsCountRef[i] << " neighborsCountProbe[" << i
                      << "]: " << neighborsCountProbe[i] << std::endl;
        }
    }
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

        CoordinateKind<double, KeyType> coords(nParticles, box, 42);

        neighborCheck(coords, radius, box);
    }
};

class FindNeighborsRandomMixD
    : public testing::TestWithParam<
          std::tuple<double, int, std::array<double, 6>, cstone::BoundaryType>>
{
public:
    template<class KeyType, template<class...> class CoordinateKind>
    void checkMixD()
    {
        double radius  = std::get<0>(GetParam());
        int nParticles = std::get<1>(GetParam());
        std::cout << "nParticles = " << nParticles << std::endl;
        auto box_limits = std::get<2>(GetParam());
        cstone::BoundaryType usePbc   = std::get<3>(GetParam());
        Box<double> box{box_limits[0], box_limits[1], box_limits[2], box_limits[3], box_limits[4],
                        box_limits[5], usePbc,        usePbc,        usePbc};

        const auto mixDBits = getBoxMixDimensionBits<double, KeyType, Box<double>>(box);

        CoordinateKind<double, KeyType> coords(nParticles, box, 42, mixDBits.bx, mixDBits.by, mixDBits.bz);

        neighborCheckMixD(coords, radius, box, mixDBits.bx, mixDBits.by, mixDBits.bz);
    }
};

class CompareNeighborsRandomMixD
    : public testing::TestWithParam<
          std::tuple<double, int, std::array<double, 6>, cstone::BoundaryType>>
{
public:
    template<class KeyType3D, class KeyTypeMixD, template<class...> class CoordinateKind>
    void compare3DtoMixD()
    {
        double radius  = std::get<0>(GetParam());
        int nParticles = std::get<1>(GetParam());
        std::cout << "nParticles = " << nParticles << std::endl;
        auto box_limits = std::get<2>(GetParam());
        cstone::BoundaryType usePbc   = std::get<3>(GetParam());
        Box<double> box{box_limits[0], box_limits[1], box_limits[2], box_limits[3], box_limits[4],
                        box_limits[5], usePbc,        usePbc,        usePbc};

        const auto mixDBits = getBoxMixDimensionBits<double, KeyTypeMixD, Box<double>>(box);

        const CoordinateKind<double, KeyType3D> coords3D(nParticles, box, 42);
        const CoordinateKind<double, KeyTypeMixD> coordsMixD(nParticles, box, 42, mixDBits.bx, mixDBits.by,
                                                       mixDBits.bz);

        unsigned bucketSize           = 64;
        auto [csTree3D, counts3D]     = computeOctree<typename KeyType3D::ValueType>(coords3D.particleKeys(), bucketSize);
        auto [csTreeMixD, countsMixD] = computeOctree<typename KeyTypeMixD::ValueType>(coordsMixD.particleKeys(), bucketSize);

        int number_of_non_zero_leaves_3D{};
        for (size_t i = 0; i < csTree3D.size(); ++i)
        {
            if (counts3D[i] > 0) { number_of_non_zero_leaves_3D++; }
        }

        int number_of_non_zero_leaves_MixD{};
        for (size_t i = 0; i < csTreeMixD.size(); ++i)
        {
            if (countsMixD[i] > 0) { number_of_non_zero_leaves_MixD++; }
        }
        std::cout << "number_of_non_zero_leaves_3D = " << number_of_non_zero_leaves_3D << std::endl;
        std::cout << "number_of_non_zero_leaves_MixD = " << number_of_non_zero_leaves_MixD << std::endl;
        EXPECT_LE(number_of_non_zero_leaves_MixD, number_of_non_zero_leaves_3D);
    }
};

TEST_P(FindNeighborsRandom, HilbertUniform32) { check<HilbertKey<uint32_t>, RandomCoordinates>(); }
TEST_P(FindNeighborsRandom, HilbertUniform64) { check<HilbertKey<uint64_t>, RandomCoordinates>(); }
TEST_P(FindNeighborsRandomMixD, HilbertMixDUniform32) { checkMixD<HilbertMixDKey<uint32_t>, RandomCoordinates>(); }
TEST_P(FindNeighborsRandomMixD, HilbertMixDUniform64) { checkMixD<HilbertMixDKey<uint64_t>, RandomCoordinates>(); }
TEST_P(CompareNeighborsRandomMixD, Hilbert3DMixDUniform32)
{
    compare3DtoMixD<HilbertKey<uint32_t>, HilbertMixDKey<uint32_t>, RandomCoordinates>();
}
TEST_P(CompareNeighborsRandomMixD, Hilbert3DMixDUniform64)
{
    compare3DtoMixD<HilbertKey<uint64_t>, HilbertMixDKey<uint64_t>, RandomCoordinates>();
}
TEST_P(FindNeighborsRandom, HilbertGaussian32) { check<HilbertKey<uint32_t>, RandomGaussianCoordinates>(); }
TEST_P(FindNeighborsRandom, HilbertGaussian64) { check<HilbertKey<uint64_t>, RandomGaussianCoordinates>(); }

std::array<double, 2> radii{0.124, 0.0624};
std::array<int, 1> nParticles{2500};
std::array<std::array<double, 6>, 2> boxes{{{0., 1., 0., 1., 0., 1.}, {-1.2, 0.23, -0.213, 3.213, -5.1, 1.23}}};
std::array<double, 3> radiiMixD{0.001, 1., 0.01};
std::array<std::array<double, 6>, 2> boxesMixD{{{0., 1., 0., 0.015625, 0., 0.00390625}, {0., 0.00390625, 0., 1., 0., 0.015625}}};
std::array<cstone::BoundaryType, 2> pbcUsage{BoundaryType::open, BoundaryType::periodic};

INSTANTIATE_TEST_SUITE_P(RandomNeighbors,
                         FindNeighborsRandom,
                         testing::Combine(testing::ValuesIn(radii),
                                          testing::ValuesIn(nParticles),
                                          testing::ValuesIn(boxes),
                                          testing::ValuesIn(pbcUsage)));

INSTANTIATE_TEST_SUITE_P(RandomNeighborsMixD,
                         FindNeighborsRandomMixD,
                         testing::Combine(testing::ValuesIn(radiiMixD),
                                          testing::ValuesIn(nParticles),
                                          testing::ValuesIn(boxesMixD),
                                          testing::ValuesIn(pbcUsage)));

INSTANTIATE_TEST_SUITE_P(CompareRandomNeighbors3DMixD,
                         CompareNeighborsRandomMixD,
                         testing::Combine(testing::ValuesIn(radiiMixD),
                                          testing::ValuesIn(nParticles),
                                          testing::ValuesIn(boxesMixD),
                                          testing::ValuesIn(pbcUsage)));

INSTANTIATE_TEST_SUITE_P(RandomNeighborsLargeRadius,
                         FindNeighborsRandom,
                         testing::Combine(testing::Values(3.0),
                                          testing::Values(500),
                                          testing::ValuesIn(boxes),
                                          testing::ValuesIn(pbcUsage)));
