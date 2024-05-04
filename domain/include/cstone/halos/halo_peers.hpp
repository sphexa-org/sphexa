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
#include "cstone/tree/definitions.h"

namespace cstone
{
namespace detail
{
inline void compactPeers(std::span<const int> flags, std::vector<int>& peers)
{
    peers.clear();
    for (int rank = 0; rank < int(flags.size()); ++rank)
    {
        if (flags[rank]) { peers.push_back(rank); };
    }
}
} // namespace detail

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
        if (focEnd < focStart) { focEnd = focStart; }

        peerFlags[rank] = layout[focEnd] > layout[focStart];
    }
    return peerFlags;
}

inline void exchangePeers(std::span<const int> exteriorPeerFlags,
                   std::vector<int>& exteriorPeers,
                   std::vector<int>& interiorPeers)
{
    std::vector<int> interiorPeerFlags(exteriorPeerFlags.size(), 0);
    MPI_Alltoall(exteriorPeerFlags.data(), 1, MPI_INT, interiorPeerFlags.data(), 1, MPI_INT, MPI_COMM_WORLD);

    detail::compactPeers(exteriorPeerFlags, exteriorPeers);
    detail::compactPeers(interiorPeerFlags, interiorPeers);
}

} // namespace cstone
