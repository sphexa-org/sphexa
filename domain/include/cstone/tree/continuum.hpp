/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Generation of octrees in cornerstone format based on particle concentration functions
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/sfc/sfc.hpp"
#include "cstone/tree/csarray.hpp"

namespace cstone
{

//! @brief estimate particle count of the given node based on the concentration continuum and node volume
template<class KeyType, class T, class F>
HOST_DEVICE_FUN unsigned continuumCount(KeyType nodeStart, KeyType nodeEnd, const Box<T>& box, F&& concentration)
{
    IBox nodeBox        = sfcIBox(sfcKey(nodeStart), sfcKey(nodeEnd));
    auto [center, size] = centerAndSize<KeyType>(nodeBox, box);

    T volume = size[0] * size[1] * size[2];

    double count = 0;
    for (int ix = -1; ix <= 1; ix += 2)
        for (int iy = -1; iy <= 1; iy += 2)
            for (int iz = -1; iz <= 1; iz += 2)
            {
                auto corner = center + 0.5 * Vec3<T>{ix * size[0], iy * size[1], iz * size[2]};
                T c         = concentration(corner[0], corner[1], corner[2]);
                count += c * volume;
            }

    return stl::min(unsigned(std::round(count)), std::numeric_limits<unsigned>::max());
}

template<class KeyType, class T, class F>
void computeContinuumCounts(
    const KeyType* tree, unsigned* counts, TreeNodeIndex numNodes, const Box<T>& box, F&& concentration)
{
#pragma ompe parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < numNodes; ++i)
    {
        counts[i] = continuumCount(tree[i], tree[i + 1], box, concentration);
    }
}

template<class F, class T, class KeyType>
bool updateContinuumCsarray(F&& concentration,
                            const Box<T>& box,
                            unsigned bucketSize,
                            std::vector<KeyType>& tree,
                            std::vector<unsigned>& counts)
{
    std::vector<TreeNodeIndex> nodeOps(nNodes(tree) + 1);
    bool converged = rebalanceDecision(tree.data(), counts.data(), nNodes(tree), bucketSize, nodeOps.data());

    std::vector<KeyType> newTree;
    rebalanceTree(tree, newTree, nodeOps.data());
    swap(tree, newTree);

    counts.resize(nNodes(tree));
    computeContinuumCounts(tree.data(), counts.data(), nNodes(tree), box, concentration);

    return converged;
}

/*! @brief Build an a cornerstone array based on continuum particle concentration
 *
 * @tparam KeyType         32- or 64-bit unsigned integer
 * @tparam F               functor object
 * @tparam T               float or double
 * @param concentration    concentration function mapping (x,y,z) in @p box to a particle concentration
 * @param box              global coordinate bounding box
 * @param bucketSize       max number of particles per (leaf) node
 * @return                 tuple(csarray, counts)
 */
template<class KeyType, class F, class T>
std::tuple<std::vector<KeyType>, std::vector<unsigned>>
computeContinuumCsarray(F&& concentration, const Box<T>& box, unsigned bucketSize)
{
    std::vector<KeyType> tree{0, nodeRange<KeyType>(0)};
    std::vector<unsigned> counts{bucketSize + 1};

    int maxIteration = 10;
    while (!updateContinuumCsarray(concentration, box, bucketSize, tree, counts) && maxIteration--)
        ;

    return std::make_tuple(std::move(tree), std::move(counts));
}

} // namespace cstone