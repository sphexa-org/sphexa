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

#include <vector>

#include "cstone/util/gsl-lite.hpp"
#include "cstone/util/pack_buffers.hpp"

namespace cstone
{

//! multiple linear buffers concatenated into a single buffer
template<class T>
class ConcatVector
{
public:
    ConcatVector() = default;
    ConcatVector(int alignment)
        : alignmentBytes_(alignment)
    {
    }

    //! @brief return number of segments
    std::size_t size() { return segments_.size(); }

    gsl::span<T> operator[](size_t i) { return segments_[i]; }
    gsl::span<const T> operator[](size_t i) const { return segments_[i]; }

    gsl::span<gsl::span<T>> view() { return segments_; }
    gsl::span<const gsl::span<const T>> cview() { return csegments_; }
    gsl::span<const gsl::span<const T>> view() const { return csegments_; }

    void reindex(std::vector<std::size_t>&& segmentSizes)
    {
        segments_  = util::packAllocBuffer<T>(buffer_, segmentSizes, alignmentBytes_);
        csegments_ = util::packAllocBuffer<const T>(buffer_, segmentSizes, alignmentBytes_);
    }

private:
    std::vector<T> buffer_;
    std::vector<gsl::span<T>> segments_;
    // Can't cast span<T>* to span<const T>*, so need to stupidly duplicate the view in return for const-correctness
    std::vector<gsl::span<const T>> csegments_;
    int alignmentBytes_{64};
};

} // namespace cstone