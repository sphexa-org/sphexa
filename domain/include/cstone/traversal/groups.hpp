/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Data structures for particle target grouping
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <utility>

#include "cstone/cuda/device_vector.h"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/tree/definitions.h"

namespace cstone
{

constexpr LocalIndex groupSize = 64;

//! @brief A view object to describe groups of target particles
/*!
 * @p activeGroups[i] stores the index of the i-th active group, relative to @p firstBody. Active group
 * @p activeGroups[i] covers the particles
 *
 *     [firstBody + activeGroups[i] * groupSize, min(firstBody + activeGroups[i] * groupSize + groupSize, lastBody)),
 *
 * i.e., all groups are blocks of groupSize consecutive particles (the last group of the range may be shorter).
 */
struct GroupView
{
    LocalIndex firstBody, lastBody;
    const LocalIndex* activeGroups = nullptr;
    LocalIndex numActiveGroups     = 0;

    constexpr LocalIndex groupStart(LocalIndex group) const
    { return firstBody + (activeGroups ? activeGroups[group] * groupSize : group * groupSize); }
    constexpr LocalIndex groupEnd(LocalIndex group) const { return std::min(groupStart(group) + groupSize, lastBody); }
    constexpr LocalIndex numGroups() const
    {
        assert((activeGroups == nullptr) == (numActiveGroups == 0));
        return activeGroups ? numActiveGroups : (lastBody - firstBody + groupSize - 1) / groupSize;
    }
};

//! @brief Describes groups of spatially close particles that can be traversed through octrees in groups
template<execution::Policy Exec>
class GroupData
{
    template<class T>
    using AccVector = std::conditional_t<execution::HaveGpu<Exec>{}, DeviceVector<T>, std::vector<T>>;

public:
    GroupData()                 = default;
    GroupData(const GroupView&) = delete;
    GroupView view() const { return {firstBody, lastBody, activeGroups, numActiveGroups}; }

    AccVector<LocalIndex> data;
    LocalIndex firstBody, lastBody;
    LocalIndex* activeGroups = nullptr;
    LocalIndex numActiveGroups;

private:
    friend void swap(GroupData& lhs, GroupData& rhs)
    {
        swap(lhs.data, rhs.data);
        std::swap(lhs.firstBody, rhs.firstBody);
        std::swap(lhs.lastBody, rhs.lastBody);
        std::swap(lhs.activeGroups, rhs.activeGroups);
        std::swap(lhs.numActiveGroups, rhs.numActiveGroups);
    }
};

} // namespace cstone
