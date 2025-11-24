/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Utilities for handling peer rank flags
 */

#pragma once

#include <algorithm>
#include <span>
#include <vector>

#include "cstone/cuda/cuda_utils.hpp"

namespace cstone
{

enum class PeerMask : int
{
    focus = 1,
    halo  = 2
};

//! @brief Return a list of ranks (peers) which contain nodes in @p focusTree that don't exist in @p globalTree
template<class KeyType>
std::vector<int> focusPeers(std::span<const TreeNodeIndex> globalOffsets,
                            std::span<const TreeIndexPair> focusOffsets,
                            int myRank,
                            std::span<const KeyType> globalTree,
                            std::span<const KeyType> focusTree)
{
    int numRanks = static_cast<int>(globalOffsets.size()) - 1;
    std::vector<int> peerFlags(numRanks, 0);
#pragma omp parallel for
    for (int rank = 0; rank < numRanks; ++rank)
    {
        if (rank == myRank) { continue; }
        auto globStart = globalTree.begin() + globalOffsets[rank];
        auto globEnd   = globalTree.begin() + globalOffsets[rank + 1];

        auto focStart = focusTree.begin() + focusOffsets[rank].start();
        auto focEnd   = focusTree.begin() + focusOffsets[rank].end();

        bool isPeer = false;
        if (focEnd - focStart > globEnd - globStart) { isPeer = true; }
        else { isPeer = not std::includes(globStart, globEnd, focStart, focEnd); }
        if (isPeer) { peerFlags[rank] |= static_cast<int>(PeerMask::focus); }
    }
    return peerFlags;
}

/*! @brief Compute list of external peers, i.e. peers from which @p myRank will request data
 *
 * @param globalOffsets  rank assignment of index ranges in the global tree
 * @param focusOffsets   rank assignment of index ranges in the focus tree (LET)
 * @param myRank
 * @param globalTree     SFC leaves of global tree, on GPU if @p useGpu == true
 * @param focusTree      SFC leaves of the focus tree, on host
 * @return               see focusPeers
 */
template<bool useGpu, class KeyType>
std::vector<int> focusPeersAcc(std::span<const TreeNodeIndex> globalOffsets,
                               std::span<const TreeIndexPair> focusOffsets,
                               int myRank,
                               std::span<const KeyType> globalTree,
                               std::span<const KeyType> focusTree)
{
    if constexpr (useGpu)
    {
        std::vector<KeyType> globalTreeBackingBuffer;
        globalTreeBackingBuffer.resize(globalTree.size());
        memcpyD2H(globalTree.data(), globalTree.size(), globalTreeBackingBuffer.data());
        auto globalTreeHost = std::span(globalTreeBackingBuffer);
        return focusPeers<KeyType>(globalOffsets, focusOffsets, myRank, globalTreeHost, focusTree);
    }
    return focusPeers<KeyType>(globalOffsets, focusOffsets, myRank, globalTree, focusTree);
}

inline void peerFlagsToList(std::span<const int> peerFlags, std::vector<int>& peersList, PeerMask mask)
{
    peersList.clear();
    for (int rank = 0; rank < int(peerFlags.size()); ++rank)
    {
        if (peerFlags[rank] & static_cast<int>(mask)) { peersList.push_back(rank); }
    }
}

} // namespace cstone
