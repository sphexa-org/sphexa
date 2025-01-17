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
#include "cstone/tree/csarray_gpu.h"
#include "cstone/tree/octree.hpp"

namespace cstone
{

/*! @brief global update step of an octree, including regeneration of the internal node structure
 *
 * @tparam        KeyType     unsigned 32- or 64-bit integer
 * @param[in]     keys    first particle key, on device
 * @param[in]     bucketSize  max number of particles per leaf
 * @param[inout]  tree        a fully linked octree
 * @param[inout]  counts      leaf node particle counts
 * @param[in]     numRanks    number of MPI ranks
 * @return                    true if tree was not changed
 */
template<class KeyType, class DevKeyVec, class DevCountVec>
bool updateOctreeGlobalGpu(std::span<const KeyType> keys,
                           unsigned bucketSize,
                           Octree<KeyType>& tree,
                           DevKeyVec& d_csTree,
                           std::vector<unsigned>& counts,
                           DevCountVec& d_counts)
{
    unsigned maxCount = std::numeric_limits<unsigned>::max();
    bool converged    = tree.rebalance(bucketSize, counts);

    counts.resize(tree.numLeafNodes());
    reallocate(d_csTree, tree.numLeafNodes() + 1, 1.01);
    reallocate(d_counts, tree.numLeafNodes(), 1.01);

    memcpyH2D(tree.treeLeaves().data(), d_csTree.size(), d_csTree.data());
    computeNodeCountsGpu(rawPtr(d_csTree), rawPtr(d_counts), tree.numLeafNodes(), keys, maxCount, true);
    memcpyD2H(d_counts.data(), d_counts.size(), counts.data());

    std::vector<unsigned> counts_reduced(counts.size());
    MPI_Allreduce(counts.data(), counts_reduced.data(), counts.size(), MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < counts.size(); ++i)
    {
        counts[i] = std::max(counts[i], counts_reduced[i]);
    }
    memcpyH2D(counts.data(), counts.size(), rawPtr(d_counts));

    return converged;
}

} // namespace cstone
