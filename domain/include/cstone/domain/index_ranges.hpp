/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Utility class to express ranges of indices
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <tuple>
#include <vector>

#include "cstone/tree/definitions.h"

namespace cstone
{

template<class T>
struct IndexPair : public std::tuple<T, T>
{
    IndexPair()
        : std::tuple<T, T>(0, 0)
    {
    }
    IndexPair(T a, T b)
        : std::tuple<T, T>(a, b)
    {
    }

    T start() const { return std::get<0>(*this); }
    T end() const { return std::get<1>(*this); }
    T count() const { return end() - start(); }
};

using TreeIndexPair = IndexPair<TreeNodeIndex>;

/*! @brief Stores ranges of local particles to be sent to another rank
 *
 * @tparam I  32- or 64-bit signed or unsigned integer to store the indices
 *
 *                                   A         B
 * Original particle buffer: |-----|****|----|****|-----|
 * Ranges:                         (6, 10)   (14,18)
 *
 * Content of offsets_: { 6, 14 }
 * Content of scan_:    { 0, 4, 8 }                                                   A    B
 * scan_ stores the offsets of a compacted buffer with the marked ranges extracted (|****|****|)
 */
template<class Index>
class IndexRanges
{
public:
    using IndexType = Index;

    IndexRanges()
        : totalCount_(0)
        , scan_{0}
    {
    }

    //! @brief add another index range, must not overlap with previous ranges
    void addRange(IndexType lower, IndexType upper)
    {
        assert(lower <= upper);
        if (lower == upper) { return; }

        totalCount_ += upper - lower;
        offsets_.push_back(lower);
        scan_.push_back(totalCount_);
    }

    [[nodiscard]] IndexType rangeStart(size_t i) const { return offsets_[i]; }

    [[nodiscard]] IndexType rangeEnd(size_t i) const { return rangeStart(i) + count(i); }

    //! @brief the number of particles in range i
    [[nodiscard]] std::size_t count(size_t i) const { return scan_[i + 1] - scan_[i]; }

    //! @brief the sum of number of particles in all ranges or total send count
    [[nodiscard]] std::size_t totalCount() const { return totalCount_; }

    [[nodiscard]] std::size_t nRanges() const { return offsets_.size(); }

    //! @brief access to arrays for upload to device
    const IndexType* offsets() const { return offsets_.data(); }
    const IndexType* scan() const { return scan_.data(); }

private:
    friend bool operator==(const IndexRanges& lhs, const IndexRanges& rhs)
    {
        return lhs.totalCount_ == rhs.totalCount_ && lhs.offsets_ == rhs.offsets_ && lhs.scan_ == rhs.scan_;
    }

    std::size_t totalCount_;
    std::vector<IndexType> offsets_;
    std::vector<IndexType> scan_;
};

//! @brief stores ranges of local particles to send out to another rank
using SendManifest = IndexRanges<LocalIndex>;

//! @brief Receive side only requires a single index range
using RecvList = std::vector<IndexPair<LocalIndex>>;

//! @brief SendList contains one manifest per rank
class SendList
{
public:
    SendList() = default;

    SendList(std::size_t numRanks)
        : data_(numRanks)
    {
    }

    SendManifest& operator[](size_t i) { return data_[i]; }
    const SendManifest& operator[](size_t i) const { return data_[i]; }

    std::size_t size() const { return data_.size(); }

    std::size_t totalCount() const
    {
        size_t count = 0;
        for (std::size_t i = 0; i < data_.size(); ++i)
        {
            count += (*this)[i].totalCount();
        }
        return count;
    }

    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }

    auto begin() const { return data_.cbegin(); }
    auto end() const { return data_.cend(); }

private:
    friend bool operator==(const SendList& lhs, const SendList& rhs) { return lhs.data_ == rhs.data_; }

    std::vector<SendManifest> data_;
};

inline size_t maxNumRanges(const SendList& sendList)
{
    size_t ret = 0;
    for (const auto& manifest : sendList)
    {
        ret = std::max(ret, manifest.nRanges());
    }
    return ret;
}

class SendRanges : public std::vector<LocalIndex>
{
    using Base = std::vector<LocalIndex>;

    size_t size() const { return Base::size(); };

public:
    SendRanges() = default;
    explicit SendRanges(int s)
        : Base(s)
    {
    }

    int numRanks() const { return int(Base::size()) - 1; }

    LocalIndex count(int rank) const
    {
        if ((*this)[rank + 1] >= (*this)[rank]) { return (*this)[rank + 1] - (*this)[rank]; }
        else { return 0; }
    }
};

//! @brief used to record or replicate a given exchange pattern or order
class ExchangeLog
{
public:
    ExchangeLog() = default;

    [[nodiscard]] bool empty() const { return log_.empty(); }

    void clear() { log_.clear(); }

    /*! @brief add a P2P message to the log
     * @param rank        destination or source rank
     * @param location    start of message in local arrays
     */
    void addExchange(int rank, LocalIndex location) { log_.emplace_back(rank, location); }

    [[nodiscard]] LocalIndex lookup(int rank) const
    {
        auto it = std::find_if(log_.begin(), log_.end(), [rank](auto e) { return std::get<0>(e) == rank; });
        assert(it != log_.end());
        return std::get<1>(*it);
    }

private:
    std::vector<std::tuple<int, LocalIndex>> log_;
};

} // namespace cstone
