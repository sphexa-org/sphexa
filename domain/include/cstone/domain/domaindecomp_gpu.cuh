/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Functions to assign a global cornerstone octree to different ranks
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * Any code in this file relies on a global cornerstone octree on each calling rank.
 */

#pragma once

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/primitives/primitives_gpu.h"
#include "domaindecomp.hpp"

namespace cstone
{

/*! @brief Based on global assignment, create the list of local particle index ranges to send to each rank
 *
 * @tparam    KeyType        32- or 64-bit integer
 * @param[in] assignment     global space curve assignment to ranks
 * @param[in] particleKeys   sorted list of SFC keys of local particles present on this rank, ON DEVICE
 * @param[-]  d_searchKeys   array of length assignment.numRanks() to store search keys, uninitialized, ON DEVICE
 * @param[-]  d_indices      array of length assignment.numRanks() to store search results, uninitialized, ON DEVICE
 * @return                    for each rank, a list of index ranges into @p particleKeys to send
 *
 * Converts the global assignment particle keys ranges into particle indices with binary search
 */
template<class KeyType>
SendRanges createSendRangesGpu(const SfcAssignment<KeyType>& assignment,
                               std::span<const KeyType> particleKeys,
                               KeyType* d_searchKeys,
                               LocalIndex* d_indices)
{
    size_t numSearchKeys = assignment.numRanks() + 1;
    SendRanges ret(numSearchKeys);

    memcpyH2D(assignment.data(), numSearchKeys, d_searchKeys);
    lowerBoundGpu(particleKeys.data(), particleKeys.data() + particleKeys.size(), d_searchKeys,
                  d_searchKeys + numSearchKeys, d_indices);
    memcpyD2H(d_indices, numSearchKeys, ret.data());

    return ret;
}

} // namespace cstone
