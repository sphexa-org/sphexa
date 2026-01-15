/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  MPI extension for calculating distributed cornerstone octrees
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <mpi.h>
#include <span>

#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/tree/csarray.hpp"
#include "cstone/tree/octree.hpp"

namespace cstone
{

/*! @brief perform one octree update, consisting of one rebalance and one node counting step
 *
 * See documentation of updateOctree
 */
template<class KeyType>
bool updateOctreeGlobal(std::span<const KeyType> keys,
                        unsigned bucketSize,
                        std::vector<KeyType>& tree,
                        std::vector<unsigned>& counts)
{
    unsigned maxCount = std::numeric_limits<unsigned>::max();

    bool converged = updateOctree(keys, bucketSize, tree, counts, maxCount);

    std::vector<unsigned> counts_reduced(counts.size());
    mpiAllreduce(counts.data(), counts_reduced.data(), counts.size(), MPI_SUM, MPI_COMM_WORLD);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < counts.size(); ++i)
    {
        counts[i] = std::max(counts[i], counts_reduced[i]);
    }

    return converged;
}

/*! @brief global update step of an octree, including regeneration of the internal node structure
 *
 * @tparam        KeyType     unsigned 32- or 64-bit integer
 * @param[in]     keys        particle SFC keys
 * @param[in]     bucketSize  max number of particles per leaf
 * @param[inout]  tree        a fully linked octree
 * @param[inout]  counts      leaf node particle counts
 * @return                    maximum number of particles per cell capped to 2^32-1 or 0 if tree has max depth
 */
template<class KeyType>
unsigned updateOctreeGlobal(std::span<const KeyType> keys,
                            unsigned bucketSize,
                            OctreeData<KeyType, CpuTag>& tree,
                            std::vector<KeyType>& leaves,
                            std::vector<unsigned>& counts)
{
    bool converged =
        rebalanceDecision(leaves.data(), counts.data(), nNodes(leaves), bucketSize, tree.childOffsets.data());
    rebalanceTree(leaves, tree.prefixes, tree.childOffsets.data());
    swap(leaves, tree.prefixes);

    tree.resize(nNodes(leaves));
    updateInternalTree<KeyType>(leaves, tree.data());

    counts.resize(tree.numLeafNodes);
    computeNodeCounts(leaves.data(), counts.data(), tree.numLeafNodes, keys, std::numeric_limits<unsigned>::max(),
                      true);

    std::vector<unsigned> counts_reduced(counts.size());
    mpiAllreduce(counts.data(), counts_reduced.data(), counts.size(), MPI_SUM, MPI_COMM_WORLD);

    unsigned maxCount = 0;
#pragma omp parallel for schedule(static) reduction(max : maxCount)
    for (size_t i = 0; i < counts.size(); ++i)
    {
        counts[i] = std::max(counts[i], counts_reduced[i]);
        maxCount  = std::max(counts[i], maxCount);
    }

    if (converged) { return 0; }
    return maxCount;
}

} // namespace cstone
