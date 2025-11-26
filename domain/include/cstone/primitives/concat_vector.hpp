/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT license
 */

/*! @file
 * @brief Specialized containers
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <span>
#include <vector>

#include "cstone/cuda/cuda_stubs.h"
#include "cstone/util/pack_buffers.hpp"

namespace cstone
{

//! multiple linear buffers concatenated into a single buffer
template<class T, template<class...> class AccVector = std::vector, int alignmentBytes = 64>
class ConcatVector
{
public:
    ConcatVector() = default;

    //! @brief return segment sizes
    std::span<const std::size_t> sizes() const { return segmentSizes_; }
    //! @brief return const span of underlying (linear) buffer
    std::span<const T> data() const { return {rawPtr(buffer_), buffer_.size()}; }

    auto view() { return util::packAllocBuffer<T>(buffer_, segmentSizes_, alignmentBytes); }
    auto view() const { return util::packAllocBuffer<const T>(buffer_, segmentSizes_, alignmentBytes); }

    auto reindex(std::vector<std::size_t>&& segmentSizes)
    {
        segmentSizes_ = std::move(segmentSizes);
        return view();
    }

private:
    friend bool operator==(const ConcatVector& lhs, const ConcatVector& rhs)
    {
        return lhs.buffer_ == rhs.buffer_ && lhs.segmentSizes_ == rhs.segmentSizes_;
    }

    mutable AccVector<T> buffer_;
    std::vector<std::size_t> segmentSizes_;
};

//! @brief copy src to dst
template<class T, template<class...> class AccVec1, template<class...> class AccVec2, int A>
void copy(const ConcatVector<T, AccVec1, A>& src, ConcatVector<T, AccVec2, A>& dst)
{
    std::vector<std::size_t> sizes(src.sizes().begin(), src.sizes().end());
    auto dstView = dst.reindex(std::move(sizes));
    T* dstBuffer = dstView.front().data();

    if constexpr (!IsDeviceVector<AccVec1<T>>{} && IsDeviceVector<AccVec2<T>>{})
    {
        memcpyH2D(src.data().data(), src.data().size(), dstBuffer);
    }
    else if constexpr (IsDeviceVector<AccVec1<T>>{} && !IsDeviceVector<AccVec2<T>>{})
    {
        memcpyD2H(src.data().data(), src.data().size(), dstBuffer);
    }
    else if constexpr (IsDeviceVector<AccVec1<T>>{} && IsDeviceVector<AccVec2<T>>{})
    {
        memcpyD2D(src.data().data(), src.data().size(), dstBuffer);
    }
    else if constexpr (!IsDeviceVector<AccVec1<T>>{} && !IsDeviceVector<AccVec2<T>>{})
    {
        std::copy_n(src.data().data(), src.data().size(), dstBuffer);
    }
}

} // namespace cstone
