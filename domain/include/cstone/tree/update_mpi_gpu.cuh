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

#include "cstone/primitives/mpi_cuda.cuh"
#include "cstone/tree/csarray_gpu.h"
#include "cstone/tree/octree_gpu.h"
#include "cstone/tree/octree.hpp"
#include "cstone/tree/update_mpi.hpp"
#include "cstone/util/pack_buffers.hpp"

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
                           OctreeData<KeyType, GpuTag>& tree,
                           std::vector<KeyType>& leaves,
                           DevKeyVec& d_csTree,
                           std::vector<unsigned>& counts,
                           DevCountVec& d_countsBuf)
{
    unsigned maxCount = std::numeric_limits<unsigned>::max();
    auto newNumNodes =
        computeNodeOpsGpu(d_csTree.data(), nNodes(d_csTree), d_countsBuf.data(), bucketSize, tree.childOffsets.data());
    reallocate(tree.prefixes, newNumNodes + 1, 1.01);
    bool converged = rebalanceTreeGpu(d_csTree.data(), nNodes(d_csTree), newNumNodes, tree.childOffsets.data(),
                                      tree.prefixes.data());
    swap(d_csTree, tree.prefixes);

    tree.resize(newNumNodes);
    buildOctreeGpu(d_csTree.data(), tree.data());

    counts.resize(tree.numLeafNodes);
    reallocate(leaves, tree.numLeafNodes + 1, 1.01);
    memcpyD2H(d_csTree.data(), d_csTree.size(), leaves.data());

    size_t numLeafNodes = tree.numLeafNodes;
    auto [d_counts, d_countsRed] =
        util::packAllocBuffer(d_countsBuf, util::TypeList<unsigned, unsigned>{}, {numLeafNodes, numLeafNodes}, 128);

    computeNodeCountsGpu(rawPtr(d_csTree), d_counts.data(), numLeafNodes, keys, maxCount, true);

    syncGpu();
    mpiAllreduceGpuDirect(d_counts.data(), d_countsRed.data(), d_counts.size(), MPI_SUM, MPI_COMM_WORLD);
    sequenceMax(d_counts.data(), d_counts.data() + d_counts.size(), d_countsRed.data(), d_counts.data());

    reallocate(counts, numLeafNodes, 1.01);
    memcpyD2H(d_counts.data(), d_counts.size(), counts.data());
    d_countsBuf.resize(numLeafNodes);

    return converged;
}

template<class KeyType, class Accelerator, class DevKeyVec, class DevCountVec>
bool updateOctreeGlobal(std::span<const KeyType> keys,
                        unsigned bucketSize,
                        OctreeData<KeyType, Accelerator>& tree,
                        std::vector<KeyType>& leaves,
                        DevKeyVec& d_csTree,
                        std::vector<unsigned>& counts,
                        DevCountVec& d_counts)
{
    if constexpr (HaveGpu<Accelerator>{})
    {
        return updateOctreeGlobalGpu(keys, bucketSize, tree, leaves, d_csTree, counts, d_counts);
    }
    else { return updateOctreeGlobal(keys, bucketSize, tree, leaves, counts); }
}

} // namespace cstone
