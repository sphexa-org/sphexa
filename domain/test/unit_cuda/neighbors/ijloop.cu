/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Neighbor loop tests
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#include <algorithm>
#include <functional>
#include <random>

#include <thrust/universal_vector.h>

#include "cstone/cuda/thrust_util.cuh"
#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/traversal/ijloop/cpu_alwaystraverse.hpp"
#include "cstone/traversal/ijloop/cpu_fullnblist.hpp"
#include "cstone/traversal/ijloop/gpu_alwaystraverse.cuh"
#include "cstone/traversal/ijloop/gpu_fullnblist.cuh"
#include "cstone/traversal/ijloop/gpu_compressednblist.cuh"
#include "cstone/traversal/ijloop/gpu_superclusternblist.cuh"

#include "../../coord_samples/random.hpp"

#include "gtest/gtest.h"

using namespace cstone;

using StrongKeyT = HilbertKey<std::uint64_t>;
using KeyT       = StrongKeyT::ValueType;

struct NeighborFun
{
    template<class ParticleData>
    constexpr auto
    operator()(ParticleData const& iData, ParticleData const& jData, Vec3<double> ijPosDiff, double distSq) const
    {
        const auto [i, iPos, hi, vi] = iData;
        const auto [j, jPos, hj, vj] = jData;
        return std::make_tuple(i, j, iPos, jPos, ijloop::symmetric::odd(ijPosDiff), ijloop::symmetric::even(distSq), hi,
                               hj, vi, vj, ijloop::symmetric::even(1u), ijloop::reduction::min(j), 37.5f);
    }
};

struct PostambleFun
{
    template<class ParticleData, class Result>
    constexpr auto operator()(ParticleData const& /* iData */, Result jResult) const
    {
        auto [iSum, jSum, iPosSum, jPosSum, ijPosDiffSum, distSqSum, hiSum, hjSum, viSum, vjSum, neighborsCount, jMin,
              constantShortSum] = jResult;
        return std::make_tuple(iSum, jSum, iPosSum, jPosSum, ijPosDiffSum, distSqSum, hiSum, hjSum, viSum, vjSum,
                               neighborsCount, hiSum / neighborsCount, jMin);
    }
};

using Result                      = std::tuple<thrust::universal_vector<LocalIndex>,   // iSum
                                               thrust::universal_vector<LocalIndex>,   // jSum
                                               thrust::universal_vector<Vec3<double>>, // iPosSum
                                               thrust::universal_vector<Vec3<double>>, // jPosSum
                                               thrust::universal_vector<Vec3<double>>, // ijPosDiffSum
                                               thrust::universal_vector<double>,       // distSqSum
                                               thrust::universal_vector<double>,       // hiSum
                                               thrust::universal_vector<double>,       // hjSum
                                               thrust::universal_vector<double>,       // viSum
                                               thrust::universal_vector<double>,       // vjSum
                                               thrust::universal_vector<unsigned>,     // neighborsCount
                                               thrust::universal_vector<double>,       // hiSumNormalized
                                               thrust::universal_vector<LocalIndex>    // jMin
                                               >;
constexpr static auto resultNames = std::make_tuple("iSum",
                                                    "jSum",
                                                    "iPosSum",
                                                    "jPosSum",
                                                    "ijPosDiffSum",
                                                    "distSqSum",
                                                    "hiSum",
                                                    "hjSum",
                                                    "viSum",
                                                    "vjSum",
                                                    "neighborsCount",
                                                    "hiSumNormalized",
                                                    "jMin");

template<class NeighborhoodBuilder>
struct IjLoopTest : testing::Test
{
    static constexpr LocalIndex totalBodies = 997, firstBody = 241, lastBody = 701;

    IjLoopTest()
    {
        RandomCoordinates<double, StrongKeyT> coords(totalBodies, box);

        x      = coords.x();
        y      = coords.y();
        z      = coords.z();
        leaves = coords.particleKeys();

        h.resize(totalBodies);
        v.resize(totalBodies);
        std::mt19937 gen(42);
        std::generate(h.begin(), h.end(), std::bind(std::uniform_real_distribution<double>(0.03, 0.15), std::ref(gen)));
        std::generate(v.begin(), v.end(), std::bind(std::uniform_real_distribution<double>(-100, 100), std::ref(gen)));

        auto [csTree, counts] = computeOctree(std::span<const KeyT>(rawPtr(leaves), leaves.size()), 8);
        OctreeData<KeyT, CpuTag> octree;
        octree.resize(nNodes(csTree));
        updateInternalTree<KeyT>(csTree, octree.data());

        layout.resize(nNodes(csTree) + 1, 0);
        std::inclusive_scan(counts.begin(), counts.end(), layout.begin() + 1);

        centers.resize(octree.numNodes), sizes.resize(octree.numNodes);
        std::span<const KeyT> nodeKeys(rawPtr(octree.prefixes), octree.numNodes);
        nodeFpCenters(nodeKeys, rawPtr(centers), rawPtr(sizes), box);

        numLeafNodes   = octree.numLeafNodes;
        numNodes       = octree.numNodes;
        prefixes       = octree.prefixes;
        childOffsets   = octree.childOffsets;
        parents        = octree.parents;
        internalToLeaf = octree.internalToLeaf;
        leafToInternal = octree.leafToInternal;
        levelRange     = octree.levelRange;

        constexpr unsigned groupSize = TravConfig::targetSize;
        const unsigned unsplitGroups = (lastBody - firstBody + groupSize - 1) / groupSize;
        groups.clear();
        for (unsigned i = 0; i < unsplitGroups; ++i)
        {
            assert(firstBody + i * groupSize < lastBody);
            groups.push_back(firstBody + i * groupSize);
            if (i == unsplitGroups / 2)
            {
                // we just split a "random" group into as-small-as-possible subgroups
                for (unsigned j = 16; j < groupSize; ++j)
                {
                    groups.push_back(firstBody + i * groupSize + j);
                }
            }
            else
            {
                // also split some other groups
                for (unsigned j : {3u, 5u, 7u})
                {
                    if (i == unsplitGroups / j) groups.push_back(firstBody + i * groupSize + j);
                }
            }
        }
        groups.push_back(lastBody);

        if (!groups.empty())
        {
            for (std::size_t g = 0; g < groups.size() - 1; ++g)
            {
                if (g % 3 == 0 || g % 5 == 0)
                {
                    subgroupStart.push_back(groups[g]);
                    subgroupEnd.push_back(groups[g + 1]);
                }
            }
        }
    }

    void setBoundaryType(BoundaryType boundaryType)
    {
        box = {box.xmin(), box.xmax(),   box.ymin(),   box.ymax(),  box.zmin(),
               box.zmax(), boundaryType, boundaryType, boundaryType};
    }

    OctreeNsView<double, KeyT> treeView(float searchExtFactor = 1.0f) const
    {
        return {.numLeafNodes    = numLeafNodes,
                .numNodes        = numNodes,
                .prefixes        = rawPtr(prefixes),
                .childOffsets    = rawPtr(childOffsets),
                .parents         = rawPtr(parents),
                .internalToLeaf  = rawPtr(internalToLeaf),
                .leafToInternal  = rawPtr(leafToInternal),
                .levelRange      = rawPtr(levelRange),
                .leaves          = rawPtr(leaves),
                .layout          = rawPtr(layout),
                .centers         = rawPtr(centers),
                .sizes           = rawPtr(sizes),
                .searchExtFactor = searchExtFactor};
    }

    GroupView groupView() const
    {
        return {.firstBody  = firstBody,
                .lastBody   = lastBody,
                .numGroups  = LocalIndex(groups.size() - 1),
                .groupStart = rawPtr(groups),
                .groupEnd   = rawPtr(groups) + 1};
    }

    GroupView subgroupView() const
    {
        return {.firstBody  = 0,
                .lastBody   = 0,
                .numGroups  = LocalIndex(subgroupStart.size()),
                .groupStart = rawPtr(subgroupStart),
                .groupEnd   = rawPtr(subgroupEnd)};
    }

    Result reference(const GroupView& groups) const
    {
        thrust::universal_vector<LocalIndex> iSum(totalBodies), jSum(totalBodies), jMin(totalBodies);
        thrust::universal_vector<Vec3<double>> iPosSum(totalBodies), jPosSum(totalBodies), ijPosDiffSum(totalBodies);
        thrust::universal_vector<double> distSqSum(totalBodies), hiSum(totalBodies), hjSum(totalBodies),
            viSum(totalBodies), vjSum(totalBodies), hiSumNormalized(totalBodies);
        thrust::universal_vector<unsigned> neighborsCount(totalBodies);

        for (unsigned g = 0; g < groups.numGroups; ++g)
        {
            const LocalIndex firstBody = groups.groupStart[g];
            const LocalIndex lastBody  = groups.groupEnd[g];
            for (unsigned i = firstBody; i < lastBody; ++i)
            {
                jMin[i] = std::numeric_limits<LocalIndex>::max();

                for (unsigned j = 0; j < totalBodies; ++j)
                {
                    const double xi = x[i];
                    const double yi = y[i];
                    const double zi = z[i];
                    const double xj = x[j];
                    const double yj = y[j];
                    const double zj = z[j];

                    double xij = xi - xj;
                    double yij = yi - yj;
                    double zij = zi - zj;

                    if (box.boundaryX() == BoundaryType::periodic)
                    {
                        if (xij < -0.5 * box.lx())
                            xij += box.lx();
                        else if (xij > 0.5 * box.lx())
                            xij -= box.lx();
                    }
                    if (box.boundaryY() == BoundaryType::periodic)
                    {
                        if (yij < -0.5 * box.ly())
                            yij += box.ly();
                        else if (yij > 0.5 * box.ly())
                            yij -= box.ly();
                    }
                    if (box.boundaryZ() == BoundaryType::periodic)
                    {
                        if (zij < -0.5 * box.lz())
                            zij += box.lz();
                        else if (zij > 0.5 * box.lz())
                            zij -= box.lz();
                    }

                    const double d2 = xij * xij + yij * yij + zij * zij;

                    if (d2 < 4 * h[i] * h[i])
                    {
                        iSum[i] += i;
                        jSum[i] += j;
                        iPosSum[i] += Vec3<double>{xi, yi, zi};
                        jPosSum[i] += Vec3<double>{xj, yj, zj};
                        ijPosDiffSum[i] += Vec3<double>{xij, yij, zij};
                        distSqSum[i] += d2;
                        hiSum[i] += h[i];
                        hjSum[i] += h[j];
                        viSum[i] += v[i];
                        vjSum[i] += v[j];
                        neighborsCount[i] += 1;
                        jMin[i] = std::min(jMin[i], LocalIndex(j));
                    }
                }
                hiSumNormalized[i] = hiSum[i] / neighborsCount[i];
            }
        }

        return {std::move(iSum),         std::move(jSum),      std::move(iPosSum),        std::move(jPosSum),
                std::move(ijPosDiffSum), std::move(distSqSum), std::move(hiSum),          std::move(hjSum),
                std::move(viSum),        std::move(vjSum),     std::move(neighborsCount), std::move(hiSumNormalized),
                std::move(jMin)};
    }

    void validate(const Result& expected, const Result& actual) const
    {

        util::for_each_tuple(
            [](auto const& e, auto const& a, const char* name)
            {
                ASSERT_EQ(e.size(), a.size());

                std::ostringstream failures;
                auto validateElem = [&failures](auto ei, auto ai, const char* name, std::size_t i)
                {
                    if constexpr (std::is_same_v<decltype(ei), double>)
                    {
                        if (std::abs(ei - ai) > 1e-8)
                            failures << "  " << name << "[" << i << "] == " << ai << " != " << ei << "\n";
                    }
                    else if constexpr (std::is_same_v<decltype(ei), Vec3<double>>)
                    {
                        if (std::abs(ei[0] - ai[0]) > 1e-8 || std::abs(ei[1] - ai[1]) > 1e-8 ||
                            std::abs(ei[2] - ai[2]) > 1e-8)
                            failures << "  " << name << "[" << i << "] == {" << ai[0] << ", " << ai[1] << ", " << ai[2]
                                     << "} != {" << ei[0] << ", " << ei[1] << ", " << ei[2] << "}\n";
                    }
                    else
                    {
                        if (ei != ai)
                            failures << "  " << name << "[" << i << "] == " << ai << " (actual) != " << ei
                                     << " (expected)\n";
                    }
                };

                for (std::size_t i = 0; i < e.size(); ++i)
                    validateElem(e[i], a[i], name, i);

                auto output = failures.view();
                if (!output.empty()) ADD_FAILURE() << output;
            },
            expected, actual, resultNames);
    }

    Box<double> box = {0, 1, BoundaryType::periodic};
    thrust::universal_vector<double> x, y, z, h, v;

    TreeNodeIndex numLeafNodes, numNodes;
    thrust::universal_vector<KeyT> leaves, prefixes;
    thrust::universal_vector<LocalIndex> layout;
    thrust::universal_vector<Vec3<double>> centers, sizes;
    thrust::universal_vector<TreeNodeIndex> childOffsets, parents, internalToLeaf, leafToInternal, levelRange;

    thrust::universal_vector<LocalIndex> groups, subgroupStart, subgroupEnd;
};

using Neighborhoods = ::testing::Types<
    ijloop::CpuAlwaysTraverseNeighborhoodBuilder,
    ijloop::CpuFullNbListNeighborhoodBuilder,
    ijloop::GpuAlwaysTraverseNeighborhoodBuilder,
    ijloop::GpuFullNbListNeighborhoodBuilder,
    ijloop::GpuCompressedNbListNeighborhoodBuilder<>::withoutSymmetry,
    ijloop::GpuCompressedNbListNeighborhoodBuilder<>::withSymmetry,
#ifdef __CUDACC__
    ijloop::GpuSuperclusterNbListNeighborhoodBuilder<>::withClusterSize<8, 4>::withoutSymmetry::withoutCompression,
    ijloop::GpuSuperclusterNbListNeighborhoodBuilder<>::withClusterSize<8, 4>::withSymmetry::withoutCompression,
    ijloop::GpuSuperclusterNbListNeighborhoodBuilder<>::withClusterSize<8, 4>::withoutSymmetry::withCompression<>,
    ijloop::GpuSuperclusterNbListNeighborhoodBuilder<>::withClusterSize<8, 4>::withSymmetry::withCompression<>,
#endif
    ijloop::GpuSuperclusterNbListNeighborhoodBuilder<>::withClusterSize<8, 8>::withoutSymmetry::withoutCompression,
    ijloop::GpuSuperclusterNbListNeighborhoodBuilder<>::withClusterSize<8, 8>::withSymmetry::withoutCompression,
    ijloop::GpuSuperclusterNbListNeighborhoodBuilder<>::withClusterSize<8, 8>::withoutSymmetry::withCompression<>,
    ijloop::GpuSuperclusterNbListNeighborhoodBuilder<>::withClusterSize<8, 8>::withSymmetry::withCompression<>>;

TYPED_TEST_SUITE(IjLoopTest, Neighborhoods);

TYPED_TEST(IjLoopTest, IjLoop)
{
    using NeighborhoodBuilder = TypeParam;

    for (BoundaryType boundaryType : {BoundaryType::open, BoundaryType::periodic, BoundaryType::fixed})
    {
        this->setBoundaryType(boundaryType);

        const auto nb =
            NeighborhoodBuilder{1024}.build(this->treeView(), this->box, this->totalBodies, this->groupView(),
                                            rawPtr(this->x), rawPtr(this->y), rawPtr(this->z), rawPtr(this->h));

        Result result;
        util::for_each_tuple([&](auto& v) { v.resize(this->totalBodies); }, result);

        auto input  = std::make_tuple(rawPtr(this->v));
        auto output = util::tupleMap([](auto& v) { return rawPtr(v); }, result);

        nb.ijLoop(input, output, NeighborFun{}, PostambleFun{});
        checkGpuErrors(cudaDeviceSynchronize());

        Result reference = this->reference(this->groupView());
        this->validate(reference, result);
    }
}

TYPED_TEST(IjLoopTest, IjLoopWithSearchExtFactor)
{
    using NeighborhoodBuilder       = TypeParam;
    constexpr float searchExtFactor = 1.5f;

    for (BoundaryType boundaryType : {BoundaryType::open, BoundaryType::periodic, BoundaryType::fixed})
    {
        this->setBoundaryType(boundaryType);

        const auto nb = NeighborhoodBuilder{1024}.build(this->treeView(searchExtFactor), this->box, this->totalBodies,
                                                        this->groupView(), rawPtr(this->x), rawPtr(this->y),
                                                        rawPtr(this->z), rawPtr(this->h));

        Result result;
        util::for_each_tuple([&](auto& v) { v.resize(this->totalBodies); }, result);

        auto input  = std::make_tuple(rawPtr(this->v));
        auto output = util::tupleMap([](auto& v) { return rawPtr(v); }, result);

        nb.ijLoop(input, output, NeighborFun{}, PostambleFun{});
        checkGpuErrors(cudaDeviceSynchronize());

        Result reference = this->reference(this->groupView());
        this->validate(reference, result);

        for (auto& h : this->h)
            h *= searchExtFactor;

        nb.ijLoop(input, output, NeighborFun{}, PostambleFun{});
        checkGpuErrors(cudaDeviceSynchronize());

        reference = this->reference(this->groupView());
        this->validate(reference, result);
    }
}

template<ijloop::NeighborhoodBuilder NeighborhoodBuilder>
consteval bool supportsSubgroup(NeighborhoodBuilder)
{
    return true;
}

template<class Config>
consteval bool supportsSubgroup(ijloop::GpuSuperclusterNbListNeighborhoodBuilder<Config>)
{
    return !Config::symmetric;
}

template<class Config>
consteval bool supportsSubgroup(ijloop::GpuCompressedNbListNeighborhoodBuilder<Config>)
{
    return false;
}

TYPED_TEST(IjLoopTest, IjLoopOnSubgroups)
{
    using NeighborhoodBuilder = TypeParam;

    if constexpr (supportsSubgroup(NeighborhoodBuilder{}))
    {
        for (BoundaryType boundaryType : {BoundaryType::open, BoundaryType::periodic, BoundaryType::fixed})
        {
            this->setBoundaryType(boundaryType);

            const auto nb =
                NeighborhoodBuilder{1024}.build(this->treeView(), this->box, this->totalBodies, this->groupView(),
                                                rawPtr(this->x), rawPtr(this->y), rawPtr(this->z), rawPtr(this->h));

            const auto subgroupNb = nb.subgroup(this->subgroupView());

            Result result;
            util::for_each_tuple([&](auto& v) { v.resize(this->totalBodies); }, result);

            auto input  = std::make_tuple(rawPtr(this->v));
            auto output = util::tupleMap([](auto& v) { return rawPtr(v); }, result);

            subgroupNb.ijLoop(input, output, NeighborFun{}, PostambleFun{});
            checkGpuErrors(cudaDeviceSynchronize());

            Result reference = this->reference(this->subgroupView());
            this->validate(reference, result);
        }
    }
    else { GTEST_SKIP() << "subgroups not supported"; }
}
