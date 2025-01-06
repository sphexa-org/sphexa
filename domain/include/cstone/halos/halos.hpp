/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Implementation of halo discovery and halo exchange
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <span>

#include "cstone/domain/exchange_keys.hpp"
#include "cstone/domain/index_ranges.hpp"
#include "cstone/halos/exchange_halos.hpp"
#ifdef USE_CUDA
#include "cstone/halos/exchange_halos_gpu.cuh"
#endif

namespace cstone
{

template<class DevVec1, class DevVec2, class... Arrays>
void haloExchangeGpu(int epoch,
                     const SendList& incomingHalos,
                     const SendList& outgoingHalos,
                     DevVec1& sendScratchBuffer,
                     DevVec2& receiveScratchBuffer,
                     Arrays... arrays);

template<class KeyType, class Accelerator>
class Halos
{
public:
    Halos(int myRank)
        : myRank_(myRank)
    {
    }

    /*! @brief Determine halo send/receive indices by informing remote notes which halos are required locally
     *
     * @param[in]  leaves      (focus) tree leaves
     * @param[in]  assignment  assignment of @p leaves to ranks
     * @param[in]  peers       list of peer ranks
     * @param[out] layout      Particle offsets for each node in @p leaves w.r.t to the final particle buffers,
     *                         including the halos, length = counts.size() + 1. The last element contains
     *                         the total number of locally present particles, i.e. assigned + halos.
     *                         [layout[i]:layout[i+1]] indexes particles in the i-th leaf cell.
     *                         If the i-th cell is not a halo and not locally owned, its particles are not present
     *                         and the corresponding layout range has length zero.
     * @return                 0 if all halo cells have been matched with a peer rank, 1 otherwise
     */
    int exchangeRequests(std::span<const KeyType> leaves,
                         std::span<const TreeIndexPair> assignment,
                         std::span<const int> peers,
                         std::span<const LocalIndex> layout)
    {
        outgoingHaloIndices_ = exchangeRequestKeys<KeyType>(leaves, assignment, peers, layout);

        incomingHaloIndices_.resize(assignment.size());
        std::fill(incomingHaloIndices_.begin(), incomingHaloIndices_.end(), RecvList::value_type{0, 0});
        for (int peer : peers)
        {
            incomingHaloIndices_[peer] = {layout[assignment[peer].start()], layout[assignment[peer].end()]};
        }

        return 0;
    }

    /*! @brief repeat the halo exchange pattern from the previous sync operation for a different set of arrays
     *
     * @param[inout] arrays  std::vector<float or double> of size particleBufferSize_
     *
     * Arrays are not resized or reallocated. Function is const, but modifies mutable haloEpoch_ counter.
     * Note that if the ScratchVectors are on device, all arrays need to be on the device too.
     */
    template<class Scratch1, class Scratch2, class... Vectors>
    void exchangeHalos(std::tuple<Vectors&...> arrays, Scratch1& sendBuffer, Scratch2& receiveBuffer) const
    {
        if constexpr (HaveGpu<Accelerator>{})
        {
            static_assert(IsDeviceVector<Scratch1>{} && IsDeviceVector<Scratch2>{});
            std::apply(
                [this, &sendBuffer, &receiveBuffer](auto&... arrays)
                {
                    haloExchangeGpu(haloEpoch_++, incomingHaloIndices_, outgoingHaloIndices_, sendBuffer, receiveBuffer,
                                    rawPtr(arrays)...);
                },
                arrays);
        }
        else
        {
            std::apply([this](auto&... arrays)
                       { haloexchange(haloEpoch_++, incomingHaloIndices_, outgoingHaloIndices_, rawPtr(arrays)...); },
                       arrays);
        }
    }

private:
    int myRank_;

    RecvList incomingHaloIndices_;
    SendList outgoingHaloIndices_;

    /*! @brief Counter for halo exchange calls
     * Multiple client calls to domain::exchangeHalos() during a time-step
     * should get different MPI tags, because there is no global MPI_Barrier or MPI collective in between them.
     */
    mutable int haloEpoch_{0};
};

} // namespace cstone
