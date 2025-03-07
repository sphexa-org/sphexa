/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Neighbor list compression
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <cassert>
#include <cstdint>
#include <iterator>

#include "cstone/cuda/annotation.hpp"
#include "cstone/primitives/clz.hpp"

namespace cstone
{

class NibbleBuffer
{
public:
    HOST_DEVICE_FUN NibbleBuffer(void* buffer, unsigned bufferSize, bool clear)
        : buffer_(reinterpret_cast<std::uint8_t*>(buffer) + sizeof(unsigned))
        , maxSize_(2 * (bufferSize - sizeof(unsigned)))
    {
        if (clear) sizeRef() = 0;
    }

    HOST_DEVICE_FUN void push_back(std::uint8_t value)
    {
        assert(value < 16);
        assert(size() < maxSize_);
        const unsigned byte   = size() / 2;
        const unsigned offset = size() % 2;
        ++sizeRef();
        buffer_[byte] = offset ? (buffer_[byte] & 0x0f) | (value << 4) : value;
    }

    HOST_DEVICE_FUN std::uint8_t pop_back()
    {
        std::uint8_t value = (*this)[size() - 1];
        --sizeRef();
        return value;
    }

    HOST_DEVICE_FUN std::uint8_t operator[](unsigned index) const
    {
        assert(index < size());
        const unsigned byte   = index / 2;
        const unsigned offset = index % 2;
        return (buffer_[byte] >> (4 * offset)) & 0xf;
    }

    HOST_DEVICE_FUN unsigned size() const { return reinterpret_cast<const unsigned*>(buffer_)[-1]; }
    HOST_DEVICE_FUN unsigned max_size() const { return maxSize_; }

    HOST_DEVICE_FUN unsigned nbytes() const { return sizeof(unsigned) + size() / 2; }

private:
    HOST_DEVICE_FUN unsigned& sizeRef() { return reinterpret_cast<unsigned*>(buffer_)[-1]; }

    std::uint8_t* buffer_;
    unsigned maxSize_;
};

class NeighborListDecompIterator;

class NeighborListCompressor
{
public:
    HOST_DEVICE_FUN NeighborListCompressor(void* buffer, unsigned bufferSize)
        : nibbles_(buffer, bufferSize, true)
    {
    }

    NeighborListCompressor(const NeighborListCompressor&)            = delete;
    NeighborListCompressor& operator=(const NeighborListCompressor&) = delete;
    NeighborListCompressor(NeighborListCompressor&&)                 = default;
    NeighborListCompressor& operator=(NeighborListCompressor&&)      = default;

    HOST_DEVICE_FUN bool push_back(std::uint32_t nbIndex)
    {
        const std::uint32_t value = nbIndex - prevNbIndex_;
        prevNbIndex_              = nbIndex;

        if (value == 1)
        {
            if (ones_) { nibbles_.push_back(nibbles_.pop_back() + 1); }
            else
            {
                if (nibbles_.size() >= nibbles_.max_size()) return false;
                nibbles_.push_back(9);
            }
            assert(nibbles_[nibbles_.size() - 1] == ones_ + 9);
            if (++ones_ >= 7) ones_ = 0;
        }
        else
        {
            const int bits    = 32 - ::countLeadingZeros(value);
            const int nibbles = (bits + 3) / 4;
            const int n       = nibbles < 1 ? 1 : nibbles;
            if (nibbles_.size() + 1 + n >= nibbles_.max_size()) return false;
            nibbles_.push_back(n);
            for (int i = 0; i < n; ++i)
                nibbles_.push_back((value >> (4 * i)) & 0xf);
            ones_ = 0;
        }
        ++nNeighbors_;
        return true;
    }

    HOST_DEVICE_FUN unsigned size() const { return nNeighbors_; }
    HOST_DEVICE_FUN unsigned nbytes() const { return nibbles_.nbytes(); }

private:
    NibbleBuffer nibbles_;
    unsigned nNeighbors_ = 0, ones_ = 0;
    std::uint32_t prevNbIndex_ = 0;
};

class NeighborListDecompressor
{
public:
    HOST_DEVICE_FUN explicit NeighborListDecompressor(const void* buffer, unsigned bufferSize)
        : nibbles_(const_cast<void*>(buffer), bufferSize, false)
    {
    }

    HOST_DEVICE_FUN unsigned nbytes() const { return nibbles_.nbytes(); }

    HOST_DEVICE_FUN NeighborListDecompIterator begin() const;
    HOST_DEVICE_FUN NeighborListDecompIterator end() const;

private:
    friend class NeighborListDecompIterator;

    NibbleBuffer nibbles_;
};

class NeighborListDecompIterator
{
public:
    using difference_type   = int;
    using value_type        = std::uint32_t;
    using pointer           = void;
    using reference         = void;
    using iterator_category = std::input_iterator_tag;

    NeighborListDecompIterator(const NeighborListDecompIterator& other)            = default;
    NeighborListDecompIterator(NeighborListDecompIterator&& other)                 = default;
    NeighborListDecompIterator& operator=(const NeighborListDecompIterator& other) = default;
    NeighborListDecompIterator& operator=(NeighborListDecompIterator&& other)      = default;

    HOST_DEVICE_FUN static NeighborListDecompIterator begin(const NeighborListDecompressor& compressor)
    {
        return {&compressor, 0};
    }
    HOST_DEVICE_FUN static NeighborListDecompIterator end(const NeighborListDecompressor& compressor)
    {
        return {&compressor, compressor.nibbles_.size()};
    }

    HOST_DEVICE_FUN bool operator==(const NeighborListDecompIterator& other) const
    {
        return i_ == other.i_ && onesLeft_ == other.onesLeft_;
    }

    HOST_DEVICE_FUN bool operator!=(const NeighborListDecompIterator& other) const { return !(*this == other); }

    HOST_DEVICE_FUN std::uint32_t operator*() const { return value_; }

    HOST_DEVICE_FUN NeighborListDecompIterator& operator++()
    {
        if (onesLeft_ > 0)
        {
            --onesLeft_;
            ++value_;
        }
        else
        {
            i_ = iNext_;
            if (i_ < decomp_->nibbles_.size()) readValue();
        }
        return *this;
    }

    HOST_DEVICE_FUN std::uint32_t operator++(int)
    {
        std::uint32_t value = **this;
        ++*this;
        return value;
    }

private:
    HOST_DEVICE_FUN NeighborListDecompIterator(const NeighborListDecompressor* compressor, unsigned i)
        : decomp_(compressor)
        , i_(i)
    {
        assert(decomp_);
        if (i == 0 && decomp_->nibbles_.size()) readValue();
    }

    HOST_DEVICE_FUN void readValue()
    {
        iNext_                 = i_;
        const int n            = decomp_->nibbles_[iNext_++];
        std::uint32_t previous = value_;
        if (n > 8)
        {
            value_    = ++previous;
            onesLeft_ = n - 9;
        }
        else
        {
            value_ = 0;
            for (int j = 0; j < n; ++j)
            {
                assert(iNext_ < decomp_->nibbles_.size());
                const std::uint32_t d = decomp_->nibbles_[iNext_++];
                value_ |= d << (4 * j);
            }
            value_ += previous;
            onesLeft_ = 0;
        }
    }

    const NeighborListDecompressor* decomp_ = nullptr;
    unsigned i_ = 0, iNext_, onesLeft_ = 0;
    std::uint32_t value_ = 0;
};

HOST_DEVICE_FUN NeighborListDecompIterator NeighborListDecompressor::begin() const
{
    return NeighborListDecompIterator::begin(*this);
}

HOST_DEVICE_FUN NeighborListDecompIterator NeighborListDecompressor::end() const
{
    return NeighborListDecompIterator::end(*this);
}

class SimpleNeighborListDecompIterator;

class SimpleNeighborListCompressor
{
public:
    HOST_DEVICE_FUN SimpleNeighborListCompressor(void* buffer, unsigned bufferSize)
        : buffer_(reinterpret_cast<std::uint32_t*>(buffer) + 2)
        , maxNonContiguous_(bufferSize / sizeof(std::uint32_t) - 2)
    {
        assert(bufferSize >= 2 * sizeof(std::uint32_t));
        sizeRef()          = 0;
        nonContiguousRef() = 0;
    }

    SimpleNeighborListCompressor(const SimpleNeighborListCompressor&)            = delete;
    SimpleNeighborListCompressor& operator=(const SimpleNeighborListCompressor&) = delete;
    SimpleNeighborListCompressor(SimpleNeighborListCompressor&&)                 = default;
    SimpleNeighborListCompressor& operator=(SimpleNeighborListCompressor&&)      = default;

    HOST_DEVICE_FUN bool push_back(std::uint32_t nbIndex)
    {
        if (nonContiguous() == 0 || nbIndex != previous_ + 1)
        {
            if (nonContiguous() >= maxNonContiguous_ || size() >= 256) return false;

            std::uint32_t diff = nbIndex - previous_;
            assert(((diff >> 23) & 0xff) == 0xff || ((diff >> 23) & 0xff) == 0x00);
            buffer_[nonContiguousRef()++] = (diff & 0x807fffff) | ((sizeRef()++ << 23) & 0x7f800000);
            previous_                     = nbIndex;
        }
        else
        {
            ++sizeRef();
            ++previous_;
        }

        return true;
    }

    HOST_DEVICE_FUN std::uint32_t size() const { return buffer_[-1]; }
    HOST_DEVICE_FUN std::uint32_t nbytes() const { return (2 + nonContiguous()) * sizeof(std::uint32_t); }

private:
    HOST_DEVICE_FUN std::uint32_t& sizeRef() { return buffer_[-1]; }
    HOST_DEVICE_FUN std::uint32_t nonContiguous() const { return buffer_[-2]; }
    HOST_DEVICE_FUN std::uint32_t& nonContiguousRef() { return buffer_[-2]; }

    std::uint32_t* buffer_;
    std::uint32_t maxNonContiguous_, previous_ = 0;
};

class SimpleNeighborListDecompressor
{
public:
    HOST_DEVICE_FUN explicit SimpleNeighborListDecompressor(const void* buffer, unsigned bufferSize)
        : buffer_(reinterpret_cast<const std::uint32_t*>(buffer) + 2)
    {
        assert(bufferSize > 2 * sizeof(std::uint32_t));
        assert(nbytes() < bufferSize);
    }

    HOST_DEVICE_FUN std::uint32_t size() const { return buffer_[-1]; }
    HOST_DEVICE_FUN unsigned nbytes() const { return (2 + nonContiguous()) * sizeof(std::uint32_t); }

    HOST_DEVICE_FUN SimpleNeighborListDecompIterator begin() const;
    HOST_DEVICE_FUN SimpleNeighborListDecompIterator end() const;

private:
    friend class SimpleNeighborListDecompIterator;

    HOST_DEVICE_FUN std::uint32_t nonContiguous() const { return buffer_[-2]; }

    std::uint32_t const* buffer_;
};

class SimpleNeighborListDecompIterator
{
public:
    using difference_type   = int;
    using value_type        = std::uint32_t;
    using pointer           = void;
    using reference         = void;
    using iterator_category = std::input_iterator_tag;

    SimpleNeighborListDecompIterator(const SimpleNeighborListDecompIterator& other)            = default;
    SimpleNeighborListDecompIterator(SimpleNeighborListDecompIterator&& other)                 = default;
    SimpleNeighborListDecompIterator& operator=(const SimpleNeighborListDecompIterator& other) = default;
    SimpleNeighborListDecompIterator& operator=(SimpleNeighborListDecompIterator&& other)      = default;

    HOST_DEVICE_FUN static SimpleNeighborListDecompIterator begin(const SimpleNeighborListDecompressor& decomp)
    {
        return {&decomp, 0};
    }
    HOST_DEVICE_FUN static SimpleNeighborListDecompIterator end(const SimpleNeighborListDecompressor& decomp)
    {
        return {&decomp, decomp.size()};
    }

    HOST_DEVICE_FUN bool operator==(const SimpleNeighborListDecompIterator& other) const { return i_ == other.i_; }

    HOST_DEVICE_FUN bool operator!=(const SimpleNeighborListDecompIterator& other) const { return !(*this == other); }

    HOST_DEVICE_FUN std::uint32_t operator*() const { return value_; }

    HOST_DEVICE_FUN SimpleNeighborListDecompIterator& operator++()
    {
        ++i_;
        readValue();
        return *this;
    }

    HOST_DEVICE_FUN std::uint32_t operator++(int)
    {
        std::uint32_t value = **this;
        ++*this;
        return value;
    }

private:
    HOST_DEVICE_FUN SimpleNeighborListDecompIterator(const SimpleNeighborListDecompressor* decomp, unsigned i)
        : decomp_(decomp)
        , i_(i)
    {
        assert(decomp_);
        readValue();
    }

    HOST_DEVICE_FUN void readValue()
    {
        while (j_ < decomp_->nonContiguous() && ((decomp_->buffer_[j_] >> 23) & 0xff) < i_)
            ++j_;

        if (j_ < decomp_->nonContiguous() && i_ == ((decomp_->buffer_[j_] >> 23) & 0xff))
        {
            std::uint32_t b = decomp_->buffer_[j_];
            value_ += b >> 31 ? (b | 0x7f800000) : (b & 0x807fffff);
        }
        else { ++value_; }
    }

    const SimpleNeighborListDecompressor* decomp_ = nullptr;
    std::uint32_t value_                          = 0;
    unsigned i_, j_ = 0;
};

HOST_DEVICE_FUN SimpleNeighborListDecompIterator SimpleNeighborListDecompressor::begin() const
{
    return SimpleNeighborListDecompIterator::begin(*this);
}

HOST_DEVICE_FUN SimpleNeighborListDecompIterator SimpleNeighborListDecompressor::end() const
{
    return SimpleNeighborListDecompIterator::end(*this);
}

} // namespace cstone
