/*! @file
 * @brief Detection and exchange of halo peer ranks
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <vector>
#include <mpi.h>
#include <span>

#include "cstone/domain/index_ranges.hpp"
#include "cstone/focus/peer_flags.hpp"
#include "cstone/tree/definitions.h"

namespace cstone
{

inline std::vector<int>
haloPeers(int myRank, std::span<const LocalIndex> layout, std::span<const TreeIndexPair> fAssignment)
{
    int numRanks = fAssignment.size();
    std::vector<int> peerFlags(numRanks, 0);
#pragma omp parallel for
    for (int rank = 0; rank < numRanks; ++rank)
    {
        if (rank == myRank) { continue; }

        TreeNodeIndex focStart = fAssignment[rank].start();
        TreeNodeIndex focEnd   = fAssignment[rank].end();

        bool isHalo = layout[focEnd] > layout[focStart];
        if (isHalo) { peerFlags[rank] |= static_cast<int>(PeerMask::halo); }
    }
    return peerFlags;
}

inline void
exchangePeers(std::span<const int> exteriorPeerFlags, std::vector<int>& exteriorPeers, std::vector<int>& interiorPeers)
{
    std::vector<int> interiorPeerFlags(exteriorPeerFlags.size(), 0);
    MPI_Alltoall(exteriorPeerFlags.data(), 1, MPI_INT, interiorPeerFlags.data(), 1, MPI_INT, MPI_COMM_WORLD);

    peerFlagsToList(exteriorPeerFlags, exteriorPeers, PeerMask::halo);
    peerFlagsToList(interiorPeerFlags, interiorPeers, PeerMask::halo);
}

} // namespace cstone
