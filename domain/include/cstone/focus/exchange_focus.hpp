/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Request counts for a locally present node structure of a remote domain from a remote rank
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * Overall procedure for each pair for peer ranks (rank1, rank2):
 *      1.  rank1 sends a node structure (vector of SFC keys) to rank2. The node structure sent by rank1
 *          covers the assigned domain of rank2. The node structure cannot exceed the resolution of
 *          the local tree of rank2, this is guaranteed by the tree-build process, as long as all ranks
 *          use the same bucket size for the locally focused tree. Usually, rank1 requests the full resolution
 *          along the surface with rank2 and a lower resolution far a way from the surface.
 *
 *      2.  rank2 receives the node structure, counts particles for each received node and sends back
 *          an answer with the particle counts per node.
 *
 *      3. rank1 receives the counts for the requested SFC keys from rank2
 */

#pragma once

#include <algorithm>
#include <vector>

#include "cstone/domain/index_ranges.hpp"
#include "cstone/primitives/concat_vector.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/primitives/mpi_cuda.cuh"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/tree/csarray.hpp"
#include "cstone/tree/csarray_gpu.h"
#include "cstone/tree/octree.hpp"
#include "cstone/util/pack_buffers.hpp"

namespace cstone
{

/*! @brief exchange subtree structures with peers
 *
 * @tparam      KeyType        32- or 64-bit unsigned integer
 * @param[in]   exteriorPeers  List of peer rank IDs to send data to
 * @param[in]   interiorPeers  List of peer rank IDs to receive data from
 * @param[in]   assignment     The assignment of @p localLeaves to peer ranks
 * @param[in]   leaves         The tree of the executing rank. Covers the global domain, but is locally focused.
 * @param[out]  treelets       The tree structures of REMOTE peer ranks covering the LOCALLY assigned part of the tree.
 *                             Each treelet covers the same SFC key range (the assigned range of the executing rank)
 *                             but is adaptively (MAC) resolved from the perspective of the peer rank.
 *
 * Note: peerTrees stores the view of REMOTE ranks for the LOCAL domain. While focusAssignment and localLeaves
 * contain the LOCAL view of REMOTE peer domains.
 */
template<class KeyType>
void exchangeTreelets(std::span<const int> exteriorPeers,
                      std::span<const int> interiorPeers,
                      std::span<const IndexPair<TreeNodeIndex>> assignment,
                      std::span<const KeyType> leaves,
                      std::vector<std::vector<KeyType>>& treelets,
                      MPI_Comm comm)
{
    constexpr int keyTag = static_cast<int>(P2pTags::focusTreelets);

    std::vector<MPI_Request> sendRequests;
    sendRequests.reserve(exteriorPeers.size());
    for (auto peer : exteriorPeers)
    {
        // +1 to include the upper key boundary for the last node
        TreeNodeIndex sendCount = assignment[peer].count() + 1;
        mpiSendAsync(leaves.data() + assignment[peer].start(), sendCount, peer, keyTag, sendRequests, comm);
    }

    std::vector<MPI_Request> receiveRequests;
    receiveRequests.reserve(interiorPeers.size());
    int numMessages = int(interiorPeers.size());
    while (numMessages--)
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, keyTag, comm, &status);
        int receiveRank = status.MPI_SOURCE;
        TreeNodeIndex receiveSize;
        MPI_Get_count(&status, MpiType<KeyType>{}, &receiveSize);
        treelets[receiveRank].resize(receiveSize);

        mpiRecvAsync(treelets[receiveRank].data(), receiveSize, receiveRank, keyTag, receiveRequests, comm);
    }

    MPI_Waitall(int(sendRequests.size()), sendRequests.data(), MPI_STATUS_IGNORE);
    MPI_Waitall(int(receiveRequests.size()), receiveRequests.data(), MPI_STATUS_IGNORE);
}

//! @brief flag treelet keys that don't exist in @p leaves as invalid
template<class KeyType>
void checkTreelets(std::span<const int> peerRanks,
                   std::span<const KeyType> leaves,
                   std::vector<std::vector<KeyType>>& treelets)
{
    for (auto rank : peerRanks)
    {
        auto& treelet          = treelets[rank];
        TreeNodeIndex numNodes = nNodes(treelets[rank]);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < numNodes; ++i)
        {
            auto k = treelet[i];
            if (k != leaves[findNodeAbove(leaves.data(), nNodes(leaves), k)]) { treelet[i] = maskKey(k); }
        }
    }
}

//! @brief remove treelet keys flagged as invalid
template<class KeyType>
void pruneTreelets(std::span<const int> interiorPeers, std::vector<std::vector<KeyType>>& treelets)
{
#pragma omp parallel for
    for (size_t r = 0; r < interiorPeers.size(); ++r)
    {
        int rank = interiorPeers[r];
        auto it  = std::remove_if(treelets[rank].begin(), treelets[rank].end(), isMasked<KeyType>);
        treelets[rank].erase(it, treelets[rank].end());
    }
}

/*! @brief exchange subtree structures with peers
 *
 * @tparam      KeyType         32- or 64-bit unsigned integer
 * @param[in]   interiorPEers   List of internal-side peer rank IDs
 * @param[in]   exteriorPEers   List of external-side peer rank IDs
 * @param[in]   leaves          leaves of the LET
 * @param[in]   treelets        The tree structures of REMOTE peer ranks covering the LOCALLY assigned part of
 *                              the tree. Each treelet covers the same SFC key range (the assigned range of
 *                              the executing rank) but is adaptively (MAC) resolved from the perspective of the
 *                              peer rank.
 * @param[out]  nodeOps         node ops needed to remove exterior keys that don't exist on the owning rank from
 *                              @p leaves
 *
 * Note: peerTrees stores the view of REMOTE ranks for the LOCAL domain. While focusAssignment and localLeaves
 * contain the LOCAL view of REMOTE peer domains.
 */
template<class KeyType>
void exchangeRejectedKeys(std::span<const int> interiorPEers,
                          std::span<const int> exteriorPEers,
                          std::span<const KeyType> leaves,
                          const std::vector<std::vector<KeyType>>& treelets,
                          std::span<TreeNodeIndex> nodeOps,
                          MPI_Comm comm)

{
    constexpr int keyTag = static_cast<int>(P2pTags::focusTreelets) + 1;

    std::vector<MPI_Request> sendRequests;
    sendRequests.reserve(interiorPEers.size());

    std::vector<std::vector<KeyType, util::DefaultInitAdaptor<KeyType>>> rejectedKeyBuffers;
    for (auto peer : interiorPEers)
    {
        auto& treelet          = treelets[peer];
        TreeNodeIndex numNodes = nNodes(treelet);

        std::vector<KeyType, util::DefaultInitAdaptor<KeyType>> rejectedKeys;
        for (int i = 0; i < numNodes; ++i)
        {
            if (isMasked(treelet[i])) { rejectedKeys.push_back(unmaskKey(treelet[i])); }
        }
        mpiSendAsync(rejectedKeys.data(), rejectedKeys.size(), peer, keyTag, sendRequests, comm);
        rejectedKeyBuffers.push_back(std::move(rejectedKeys));
    }

    int numMessages = exteriorPEers.size();
    while (numMessages--)
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, keyTag, comm, &status);
        int receiveRank = status.MPI_SOURCE;
        TreeNodeIndex receiveSize;
        MPI_Get_count(&status, MpiType<KeyType>{}, &receiveSize);

        std::vector<KeyType, util::DefaultInitAdaptor<KeyType>> recvKeys(receiveSize);
        recvKeys.resize(receiveSize);
        mpiRecvSync(recvKeys.data(), receiveSize, receiveRank, keyTag, &status, comm);
        for (TreeNodeIndex i = 0; i < receiveSize; ++i)
        {
            TreeNodeIndex ki = findNodeAbove(leaves.data(), leaves.size(), recvKeys[i]);
            nodeOps[ki]      = 0;
        }
    }

    MPI_Waitall(int(sendRequests.size()), sendRequests.data(), MPI_STATUS_IGNORE);
}

template<class KeyType>
void syncTreelets(std::span<const int> exteriorPeers,
                  std::span<const int> interiorPeers,
                  std::span<const IndexPair<TreeNodeIndex>> assignment,
                  OctreeData<KeyType, CpuTag>& octree,
                  std::vector<KeyType>& leaves,
                  std::vector<std::vector<KeyType>>& treelets,
                  MPI_Comm comm)
{
    exchangeTreelets<KeyType>(exteriorPeers, interiorPeers, assignment, leaves, treelets, comm);
    checkTreelets<KeyType>(interiorPeers, leaves, treelets);

    std::vector<TreeNodeIndex> nodeOps(leaves.size(), 1);
    exchangeRejectedKeys<KeyType>(interiorPeers, exteriorPeers, leaves, treelets, nodeOps, comm);
    pruneTreelets<KeyType>(interiorPeers, treelets);

    if (std::count(nodeOps.begin(), nodeOps.end(), 1) != std::make_signed_t<size_t>(nodeOps.size()))
    {
        rebalanceTree(leaves, octree.prefixes, nodeOps.data());
        swap(leaves, octree.prefixes);
        octree.resize(nNodes(leaves));
        updateInternalTree<KeyType>(leaves, octree.data());
    }
}

template<class KeyType, class Vector>
void syncTreeletsGpu(std::span<const int> exteriorPeers,
                     std::span<const int> interiorPeers,
                     std::span<const IndexPair<TreeNodeIndex>> assignment,
                     const std::vector<KeyType>& leaves,
                     OctreeData<KeyType, GpuTag>& octreeAcc,
                     DeviceVector<KeyType>& leavesAcc,
                     std::vector<std::vector<KeyType>>& treelets,
                     Vector& scratch,
                     MPI_Comm comm)
{
    exchangeTreelets<KeyType>(exteriorPeers, interiorPeers, assignment, leaves, treelets, comm);
    checkTreelets<KeyType>(interiorPeers, leaves, treelets);

    std::vector<TreeNodeIndex> nodeOps(leaves.size(), 1);
    exchangeRejectedKeys<KeyType>(interiorPeers, exteriorPeers, leaves, treelets, nodeOps, comm);
    pruneTreelets<KeyType>(interiorPeers, treelets);

    if (std::count(nodeOps.begin(), nodeOps.end(), 1) != long(nodeOps.size()))
    {
        assert(octreeAcc.childOffsets.size() >= nodeOps.size());
        std::span<TreeNodeIndex> nops(rawPtr(octreeAcc.childOffsets), nodeOps.size());
        memcpyH2D(rawPtr(nodeOps), nodeOps.size(), nops.data());

        exclusiveScanGpu(nops.data(), nops.data() + nops.size(), nops.data());
        TreeNodeIndex newNumLeafNodes;
        memcpyD2H(nops.data() + nops.size() - 1, 1, &newNumLeafNodes);

        auto& newLeaves = octreeAcc.prefixes;
        reallocateDestructive(newLeaves, newNumLeafNodes + 1, 1.05);
        rebalanceTreeGpu(rawPtr(leavesAcc), nNodes(leavesAcc), newNumLeafNodes, nops.data(), rawPtr(newLeaves));
        swap(newLeaves, leavesAcc);

        octreeAcc.resize(nNodes(leavesAcc));

        size_t newNumNodes        = octreeAcc.numNodes;
        size_t spaceForLevelRange = sizeof(TreeNodeIndex) * (maxTreeLevel<KeyType>{} + 2);
        size_t cubTmpSize = std::max(sortByKeyTempStorage<KeyType, TreeNodeIndex>(newNumNodes), spaceForLevelRange);

        auto originalSize               = scratch.size();
        auto [keyBuf, valueBuf, cubTmp] = util::packAllocBuffer(scratch, util::TypeList<KeyType, TreeNodeIndex, char>{},
                                                                {newNumNodes, newNumNodes, cubTmpSize}, 128);

        buildOctreeGpu(rawPtr(leavesAcc), octreeAcc.data(), keyBuf, valueBuf, cubTmp);
        scratch.resize(originalSize);
    }
}

template<class VecOfVec>
std::vector<std::size_t> extractNumNodes(const VecOfVec& vov)
{
    std::vector<std::size_t> ret(vov.size());
    for (size_t i = 0; i < vov.size(); ++i)
    {
        ret[i] = vov[i].empty() ? 0 : nNodes(vov[i]);
    }
    return ret;
}

//! @brief assign treelet nodes their final indices w.r.t the final LET
template<class KeyType>
void indexTreelets(std::span<const int> peerRanks,
                   std::span<const KeyType> nodeKeys,
                   std::span<const TreeNodeIndex> levelRange,
                   const std::vector<std::vector<KeyType>>& treelets,
                   ConcatVector<TreeNodeIndex>& treeletIdx)
{
    auto tlView = treeletIdx.reindex(extractNumNodes(treelets));
    for (int rank : peerRanks)
    {
        const auto& treelet    = treelets[rank];
        auto tlIdx             = tlView[rank];
        TreeNodeIndex numNodes = nNodes(treelets[rank]);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < numNodes; ++i)
        {
            tlIdx[i] = locateNode(treelet[i], treelet[i + 1], nodeKeys.data(), levelRange.data());
            assert(tlIdx[i] < TreeNodeIndex(nodeKeys.size()));
        }
    }
}

//! @brief send cell properties, send to interior peers, recv from exterior peers
template<class T, class DevVec>
void exchangeTreeletGeneral(std::span<const int> interiorPeers,
                            std::span<const int> exteriorPeers,
                            std::span<const std::span<const TreeNodeIndex>> treeletIdx,
                            std::span<const IndexPair<TreeNodeIndex>> focusAssignment,
                            std::span<const TreeNodeIndex> csToInternalMap,
                            std::span<T> quantities,
                            int commTag,
                            DevVec& scratch,
                            MPI_Comm comm)
{
    constexpr int alignmentBytes = 64;
    constexpr bool useGpu        = IsDeviceVector<DevVec>{};

    std::vector<std::size_t> treeletSizes(interiorPeers.size() + exteriorPeers.size());
    for (size_t i = 0; i < interiorPeers.size(); ++i)
        treeletSizes[i] = treeletIdx[interiorPeers[i]].size(); // send buffers
    for (size_t i = 0; i < exteriorPeers.size(); ++i)
        treeletSizes[i + interiorPeers.size()] = focusAssignment[exteriorPeers[i]].count(); // recv buffers

    size_t origSize    = scratch.size();
    auto packedBuffers = util::packAllocBuffer<T>(scratch, treeletSizes, alignmentBytes);
    std::span<std::span<T>> sendBuffers{packedBuffers.data(), interiorPeers.size()};
    std::span<std::span<T>> recvBuffers{packedBuffers.data() + interiorPeers.size(), exteriorPeers.size()};

    std::vector<std::vector<T, util::DefaultInitAdaptor<T>>> staging; // only used if GPU-direct is not active
    std::vector<MPI_Request> sendRequests;
    sendRequests.reserve(interiorPeers.size());
    for (size_t i = 0; i < interiorPeers.size(); ++i)
    {
        gatherAcc<useGpu, TreeNodeIndex>(treeletIdx[interiorPeers[i]], quantities.data(), sendBuffers[i].data());
        if constexpr (useGpu) { syncGpu(); }
        assert(sendBuffers[i].size() == treeletIdx[interiorPeers[i]].size());
        mpiSendAsyncAcc<useGpu>(sendBuffers[i].data(), treeletIdx[interiorPeers[i]].size(), interiorPeers[i], commTag,
                                sendRequests, staging, comm);
    }

    int numMessages = exteriorPeers.size();
    while (numMessages--)
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, commTag, comm, &status);
        int recvRank = status.MPI_SOURCE;
        TreeNodeIndex recvCount;
        mpiGetCount<T>(&status, &recvCount);

        int peerIdx = std::find(exteriorPeers.begin(), exteriorPeers.end(), recvRank) - exteriorPeers.begin();
        T* recvBuf  = recvBuffers[peerIdx].data();
        mpiRecvSyncAcc<useGpu>(recvBuf, recvCount, recvRank, commTag, MPI_STATUS_IGNORE, comm);

        auto mapToInternal = csToInternalMap.subspan(focusAssignment[recvRank].start(), recvCount);
        scatterAcc<useGpu>(mapToInternal, recvBuf, quantities.data());
    }
    if constexpr (useGpu) { syncGpu(); }

    MPI_Waitall(int(sendRequests.size()), sendRequests.data(), MPI_STATUS_IGNORE);
    reallocate(scratch, origSize, 1.0);
}

} // namespace cstone
