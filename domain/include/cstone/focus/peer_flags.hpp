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
std::vector<int> focusPeers(std::span<const KeyType> boundaries,
                            int myRank,
                            std::span<const KeyType> globalTree,
                            std::span<const KeyType> focusTree)
{
    int numRanks = static_cast<int>(boundaries.size()) - 1;
    std::vector<int> peerFlags(numRanks, 0);
#pragma omp parallel for
    for (int rank = 0; rank < numRanks; ++rank)
    {
        if (rank == myRank) { continue; }
        auto globStart = std::lower_bound(globalTree.begin(), globalTree.end(), boundaries[rank]);
        auto globEnd   = std::lower_bound(globalTree.begin(), globalTree.end(), boundaries[rank + 1]);

        auto focStart = std::lower_bound(focusTree.begin(), focusTree.end(), boundaries[rank]);
        auto focEnd   = std::upper_bound(focusTree.begin(), focusTree.end(), boundaries[rank + 1]) - 1;
        if (focEnd < focStart) { focEnd = focStart; }

        bool isPeer = false;
        if (focEnd - focStart > globEnd - globStart) { isPeer = true; }
        else { isPeer = not std::includes(globStart, globEnd, focStart, focEnd); }
        if (isPeer) { peerFlags[rank] |= static_cast<int>(PeerMask::focus); }
    }
    return peerFlags;
}

/*! @brief Compute list of external peers, i.e. peers from which @p myRank will request data
 *
 * @param boundaries  SFC start key of each rank/subdomain
 * @param myRank
 * @param globalTree  SFC leaves of global tree, on GPU if @p useGpu == true
 * @param focusTree   SFC leaves of the focus tree, on host
 * @return            see focusPeers
 */
template<bool useGpu, class KeyType>
std::vector<int> focusPeersAcc(std::span<const KeyType> boundaries,
                               int myRank,
                               std::span<const KeyType> globalTree,
                               std::span<const KeyType> focusTree)
{
    auto globalTreeActive = globalTree;
    if constexpr (useGpu)
    {
        std::vector<KeyType> globalTreeBackingBuffer;
        globalTreeBackingBuffer.resize(globalTree.size());
        memcpyD2H(globalTree.data(), globalTree.size(), globalTreeBackingBuffer.data());
        globalTreeActive = std::span(globalTreeBackingBuffer);
    }
    return focusPeers<KeyType>(boundaries, myRank, globalTreeActive, focusTree);
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
