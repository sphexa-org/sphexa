/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Send/receive SFC key ranges to let peer ranks know which particles to send as halos
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/domain/layout.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"

namespace cstone
{

/*! @brief exchange halo request keys, establish particle indices to send
 *
 * @tparam KeyType      32- or 64-bit unsigned integer
 * @param treeLeaves    cornerstone octree leaves, length N+1
 * @param assignment    assignment of @p treeLeaves to ranks, only ranks listed in @p peerRanks
 *                      are accessed
 * @param peerRanks     list of peer rank IDs
 * @param layout        particle location (index in buffers) of each node in @a treeLeaves
 * @return              a SendList, containing ranges of local particle indices to send out
 *                      to each peer rank in subsequent halo particle exchanges.
 *
 * Preconditions:
 *
 *    Let node index i on rank r1 be marked as a halo, i.e. haloFlags[i] == 1.
 *    - Then the assigned rank of i, r2 = assignment.findRank(i) has to be listed in @p peerRanks,
 *      otherwise the request is not sent and r1 will not receive those halos in the subsequent
 *      halo particle exchange, which will cause it to fail.
 *    - r1 sends the range [treeLeaves[i]:treeLeaves[i+1]] to r2, so that range has to be part of
 *      the assignment of r2.
 */
template<class KeyType>
SendList exchangeRequestKeys(std::span<const KeyType> treeLeaves,
                             std::span<const TreeIndexPair> assignment,
                             std::span<const int> peerRanks,
                             std::span<const LocalIndex> layout)
{
    std::vector<std::vector<KeyType>> sendBuffers;
    sendBuffers.reserve(peerRanks.size());

    std::vector<MPI_Request> sendRequests;

    constexpr int haloRequestKeyTag = static_cast<int>(P2pTags::haloRequestKeys);

    for (int peer : peerRanks)
    {
        auto requestKeys = extractMarkedElements(treeLeaves, layout, assignment[peer].start(), assignment[peer].end());
        mpiSendAsync(requestKeys.data(), int(requestKeys.size()), peer, haloRequestKeyTag, sendRequests);
        sendBuffers.push_back(std::move(requestKeys));
    }

    std::vector<KeyType> receiveBuffer(treeLeaves.size());

    SendList ret(assignment.size());

    size_t numMessages = peerRanks.size();
    while (numMessages > 0)
    {
        MPI_Status status;
        mpiRecvSync(receiveBuffer.data(), receiveBuffer.size(), MPI_ANY_SOURCE, haloRequestKeyTag, &status);
        int receiveRank = status.MPI_SOURCE;
        TreeNodeIndex numKeys;
        MPI_Get_count(&status, MpiType<KeyType>{}, &numKeys);

        for (TreeNodeIndex i = 0; i < numKeys; i += 2)
        {
            KeyType lowerKey = receiveBuffer[i];
            KeyType upperKey = receiveBuffer[i + 1];

            LocalIndex lowerIdx = layout[findNodeAbove(treeLeaves.data(), treeLeaves.size(), lowerKey)];
            LocalIndex upperIdx = layout[findNodeAbove(treeLeaves.data(), treeLeaves.size(), upperKey)];

            ret[receiveRank].addRange(lowerIdx, upperIdx);
        }

        numMessages--;
    }

    MPI_Status status[sendRequests.size()];
    MPI_Waitall(int(sendRequests.size()), sendRequests.data(), status);

    // MUST call MPI_Barrier or any other collective MPI operation that enforces synchronization
    // across all ranks before calling this function again
    // MPI_Barrier(MPI_COMM_WORLD);

    return ret;
}

} // namespace cstone
