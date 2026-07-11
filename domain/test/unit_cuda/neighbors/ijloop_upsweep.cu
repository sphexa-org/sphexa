/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Tree upsweep test
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#include <span>

#include <thrust/universal_vector.h>

#include "cstone/cuda/stream_holder.cuh"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/traversal/ijloop/upsweep.cuh"
#include "cstone/util/tuple_util.hpp"

#include "../../coord_samples/random.hpp"

#include "gtest/gtest.h"

using namespace cstone;

using StrongKeyT = HilbertKey<std::uint64_t>;
using KeyT       = StrongKeyT::ValueType;

template<class Tc, class KeyType, class TransformOp, class BinaryOp, class... In, class... Out>
void upsweepReference(const OctreeNsView<Tc, KeyType>& tree,
                      const std::tuple<Out...>& init,
                      TransformOp&& transformOp,
                      BinaryOp&& binaryOp,
                      const std::tuple<In*...>& input,
                      std::tuple<Out*...> output)
{
    auto numInternalNodes = tree.numNodes - tree.numLeafNodes;
#pragma omp parallel for
    for (TreeNodeIndex leafIdx = 0; leafIdx < tree.numLeafNodes; ++leafIdx)
    {
        TreeNodeIndex nodeIdx = tree.leafToInternal[numInternalNodes + leafIdx];
        auto accum            = init;
        for (LocalIndex i = tree.layout[leafIdx]; i < tree.layout[leafIdx + 1]; ++i)
            accum = binaryOp(accum, transformOp(util::tupleMap([&](const auto* ptr) { return ptr[i]; }, input)));
        util::for_each_tuple([&](auto* ptr, auto value) { ptr[nodeIdx] = value; }, output, accum);
    }

    for (int currentLevel = maxTreeLevel<KeyType>(); currentLevel >= 0; --currentLevel)
    {
        const TreeNodeIndex start = tree.levelRange[currentLevel];
        const TreeNodeIndex end   = tree.levelRange[currentLevel + 1];
#pragma omp parallel for
        for (TreeNodeIndex nodeIdx = start; nodeIdx < end; ++nodeIdx)
        {
            const TreeNodeIndex firstChild = tree.childOffsets[nodeIdx];
            if (firstChild)
            {
                auto accum = init;
                for (TreeNodeIndex childIdx = firstChild; childIdx < firstChild + eightSiblings; ++childIdx)
                    accum = binaryOp(accum, util::tupleMap([&](const auto* ptr) { return ptr[childIdx]; }, output));

                util::for_each_tuple([&](auto* ptr, auto value) { ptr[nodeIdx] = value; }, output, accum);
            }
        }
    }
}

struct TransformOp
{
    constexpr std::tuple<long> operator()(const std::tuple<long, bool>& value) const
    {
        const auto [v, inv] = value;
        return {inv ? -v : v};
    }
};

struct BinaryOp
{
    constexpr std::tuple<long> operator()(const std::tuple<long>& accum, const std::tuple<long>& value) const
    {
        return util::tupleMap(std::plus<void>(), accum, value);
    }
};

TEST(IjLoop, Upsweep)
{
    const unsigned totalBodies = 997;
    Box<double> box{0, 1, BoundaryType::periodic};
    RandomCoordinates<double, StrongKeyT> coords(totalBodies, box);

    thrust::universal_vector<double> x  = coords.x();
    thrust::universal_vector<double> y  = coords.y();
    thrust::universal_vector<double> z  = coords.z();
    thrust::universal_vector<KeyT> keys = coords.particleKeys();

    thrust::universal_vector<double> h(totalBodies), v(totalBodies);
    std::mt19937 gen(42);
    std::generate(h.begin(), h.end(), std::bind(std::uniform_real_distribution<double>(0.03, 0.15), std::ref(gen)));
    std::generate(v.begin(), v.end(), std::bind(std::uniform_real_distribution<double>(-100, 100), std::ref(gen)));

    auto [csTree, counts] = computeOctree(std::span<const KeyT>(rawPtr(keys), keys.size()), 8);
    OctreeData<KeyT, execution::Cpu> octree;
    octree.resize(nNodes(csTree));
    updateInternalTree<KeyT>(csTree, octree.data());

    thrust::universal_vector<LocalIndex> layout(nNodes(csTree) + 1, 0);
    std::inclusive_scan(counts.begin(), counts.end(), layout.begin() + 1);

    thrust::universal_vector<Vec3<double>> centers(octree.numNodes), sizes(octree.numNodes);
    std::span<const KeyT> nodeKeys(rawPtr(octree.prefixes), octree.numNodes);
    nodeFpCenters(nodeKeys, rawPtr(centers), rawPtr(sizes), box);

    thrust::universal_vector<KeyT> octreePrefixes              = octree.prefixes;
    thrust::universal_vector<TreeNodeIndex> octreeChildOffsets = octree.childOffsets, octreeParents = octree.parents,
                                            octreeInternalToLeaf = octree.internalToLeaf,
                                            octreeLeafToInternal = octree.leafToInternal,
                                            octreeLevelRange     = octree.levelRange;

    OctreeNsView<double, KeyT> view{.numLeafNodes   = octree.numLeafNodes,
                                    .numNodes       = octree.numNodes,
                                    .prefixes       = rawPtr(octreePrefixes),
                                    .childOffsets   = rawPtr(octreeChildOffsets),
                                    .parents        = rawPtr(octreeParents),
                                    .internalToLeaf = rawPtr(octreeInternalToLeaf),
                                    .leafToInternal = rawPtr(octreeLeafToInternal),
                                    .levelRange     = rawPtr(octreeLevelRange),
                                    .leaves         = rawPtr(keys),
                                    .layout         = rawPtr(layout),
                                    .centers        = rawPtr(centers),
                                    .sizes          = rawPtr(sizes)};

    thrust::universal_vector<long> dataLong(totalBodies);
    thrust::universal_vector<bool> dataBool(totalBodies);
    std::generate(dataLong.begin(), dataLong.end(),
                  std::bind(std::uniform_int_distribution<long>(-100, 100), std::ref(gen)));
    std::generate(dataBool.begin(), dataBool.end(), std::bind(std::bernoulli_distribution(), std::ref(gen)));

    thrust::universal_vector<long> reference(octree.numNodes);
    upsweepReference(view, std::tuple(0l), TransformOp(), BinaryOp(), std::tuple(rawPtr(dataLong), rawPtr(dataBool)),
                     std::tuple(rawPtr(reference)));

    thrust::universal_vector<long> result(octree.numNodes);
    StreamHolder stream;
    ijloop::upsweep(stream.exec(), view, std::tuple(0l), TransformOp(), BinaryOp(),
                    std::tuple(rawPtr(dataLong), rawPtr(dataBool)), std::tuple(rawPtr(result)));
    stream.sync();

    EXPECT_EQ(result, reference);
}
