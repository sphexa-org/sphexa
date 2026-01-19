/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Neighbor list compression
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

// if 1, use compression proposed in Compressed Neighbour Lists for SPH, by S. Band, C. Gissler and M. Teschner, 2020
#ifndef CSTONE_USE_BAND_ET_AL_COMPRESSION
#define CSTONE_USE_BAND_ET_AL_COMPRESSION 0
#endif

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "cstone/cuda/gpu_config.cuh"
#include "cstone/primitives/clz.hpp"
#include "cstone/primitives/warpscan.cuh"

namespace cstone
{

struct WarpCompression
{
    __device__ __forceinline__ explicit WarpCompression(void* output)
        : start_(reinterpret_cast<std::uint8_t*>(output))
        , buffer_(reinterpret_cast<std::uint8_t*>(reinterpret_cast<unsigned*>(output) + 1))
        , previous_(static_cast<unsigned>(-1))
        , numNeighbors_(0)
    {
    }

    WarpCompression(const WarpCompression&)            = delete;
    WarpCompression& operator=(const WarpCompression&) = delete;
    WarpCompression(WarpCompression&&)                 = delete;
    WarpCompression& operator=(WarpCompression&&)      = delete;

    __device__ __forceinline__ void add(const unsigned neighbor, const unsigned activeLanes)
    {
        assert(activeLanes > 0);

        numNeighbors_ += activeLanes;
        const unsigned laneIdx = laneIndex();

#if CSTONE_USE_BAND_ET_AL_COMPRESSION
        std::uint8_t* vleData = buffer_ + 2 * sizeof(GpuConfig::ThreadMask);

        const unsigned leftNeighbor = shflUpSync(neighbor, 1);
        const unsigned diff = laneIdx < activeLanes ? (neighbor - (laneIdx > 0 ? leftNeighbor : previous_)) - 1 : 0;
        previous_           = shflSync(neighbor, GpuConfig::warpSize - 1);

        const auto firstControl   = ballotSync(diff > 1);
        const auto secondControl  = ballotSync((diff == 1) | (diff >= 256));
        const auto controlToStore = laneIdx < sizeof(GpuConfig::ThreadMask) ? firstControl : secondControl;
        if (laneIdx < 2 * sizeof(GpuConfig::ThreadMask))
            buffer_[laneIdx] = controlToStore >> (8 * (laneIdx % sizeof(GpuConfig::ThreadMask)));

        const unsigned dataBytes      = diff >= 2 ? (diff >= 256 ? 4 : 1) : 0;
        const unsigned dataBytesScan  = inclusiveScanInt(dataBytes);
        const unsigned dataBytesIndex = dataBytesScan - dataBytes;
        const unsigned warpDataBytes  = shflSync(dataBytesScan, GpuConfig::warpSize - 1);

        for (unsigned i = 0; i < dataBytes; ++i)
            vleData[dataBytesIndex + i] = (diff >> (8 * i)) & 0xff;

        buffer_ += 2 * sizeof(GpuConfig::ThreadMask) + warpDataBytes;
#else
        std::uint8_t* vleData = buffer_ + sizeof(GpuConfig::ThreadMask);

        const auto writeDataNibble = [&](unsigned index, std::uint8_t value, bool odd)
        {
            assert(value < 16);
            if (odd == index % 2)
            {
                std::uint8_t byte = odd ? vleData[index / 2] : 0;
                byte |= (value << ((index % 2) * 4));
                vleData[index / 2] = byte;
            }
        };

        const unsigned leftNeighbor = shflUpSync(neighbor, 1);
        const unsigned diff         = neighbor - (laneIdx > 0 ? leftNeighbor : previous_);
        previous_                   = shflSync(neighbor, GpuConfig::warpSize - 1);

        const bool nonOne     = (diff != 1) & (laneIdx < activeLanes);
        const auto nonOneBits = ballotSync(nonOne);
        if (laneIdx < sizeof(GpuConfig::ThreadMask)) buffer_[laneIdx] = (nonOneBits >> (8 * laneIdx));
        const bool additionalStorage = (diff > 9) & (laneIdx < activeLanes);
        const unsigned nBits         = 32 - countLeadingZeros(diff);
        const unsigned nNibbles      = additionalStorage ? (nBits + 3) / 4 : 0;
        const unsigned nNibblesData  = additionalStorage ? nNibbles - 1 : diff + 6;

        const unsigned nNibblesIndex = popCount(nonOneBits & lanemask_lt());
        unsigned vleDataSize         = popCount(nonOneBits);

        if (nonOne) writeDataNibble(nNibblesIndex, nNibblesData, false);
#ifdef __HIP_PLATFORM_AMD__
        // This should not be necessary, a memory fence should be enough, but tests fail without
        __syncthreads();
#else
        syncWarp();
#endif
        if (nonOne) writeDataNibble(nNibblesIndex, nNibblesData, true);

        const unsigned nbValueScan      = inclusiveScanInt(nNibbles);
        const unsigned nbValueDataIndex = vleDataSize + nbValueScan - nNibbles;
        const unsigned nbValueSize      = shflSync(nbValueScan, GpuConfig::warpSize - 1);
        vleDataSize += nbValueSize;

        for (unsigned i = 0; i < nNibbles; ++i)
            writeDataNibble(nbValueDataIndex + i, (diff >> (4 * i)) & 0xf, false);
        syncWarp();
        for (unsigned i = 0; i < nNibbles; ++i)
            writeDataNibble(nbValueDataIndex + i, (diff >> (4 * i)) & 0xf, true);

        buffer_ += sizeof(GpuConfig::ThreadMask) + (vleDataSize + 1) / 2;
#endif
    }

    __device__ __forceinline__ unsigned numBytes() const { return (unsigned)(buffer_ - (std::uint8_t*)start_); }
    __device__ __forceinline__ unsigned numNeighbors() const { return numNeighbors_; }

    __device__ __forceinline__ ~WarpCompression()
    {
        const unsigned laneIdx    = laneIndex();
        const unsigned totalBytes = numBytes();
        assert(numNeighbors_ > 0 || totalBytes == sizeof(unsigned));
        if (laneIdx == 0) *reinterpret_cast<unsigned*>(start_) = totalBytes;
    }

private:
    std::uint8_t *start_, *buffer_;
    unsigned previous_, numNeighbors_;
};

struct WarpDecompression
{
    __device__ __forceinline__ explicit WarpDecompression(const void* input, const unsigned numNeighbors)
        : buffer_(reinterpret_cast<const std::uint8_t*>(reinterpret_cast<const unsigned*>(input) + 1))
        , previous_(static_cast<unsigned>(-1))
        , numBytes_(*reinterpret_cast<const unsigned*>(input))
        , numNeighbors_(numNeighbors)
    {
    }

    WarpDecompression(const WarpDecompression&)            = delete;
    WarpDecompression& operator=(const WarpDecompression&) = delete;
    WarpDecompression(WarpDecompression&&)                 = delete;
    WarpDecompression& operator=(WarpDecompression&&)      = delete;

    __device__ __forceinline__ unsigned numBytes() const { return numBytes_; }
    __device__ __forceinline__ unsigned numNeighbors() const { return numNeighbors_; }

    __device__ __forceinline__ unsigned next()
    {
        const unsigned laneIdx = laneIndex();

#if CSTONE_USE_BAND_ET_AL_COMPRESSION
        const std::uint8_t* vleData = buffer_ + 2 * sizeof(GpuConfig::ThreadMask);

        GpuConfig::ThreadMask firstControl  = 0;
        GpuConfig::ThreadMask secondControl = 0;
        for (unsigned i = 0; i < sizeof(GpuConfig::ThreadMask); ++i)
        {
            firstControl |= GpuConfig::ThreadMask(buffer_[i]) << (8 * i);
            secondControl |= GpuConfig::ThreadMask(buffer_[sizeof(GpuConfig::ThreadMask) + i]) << (8 * i);
        }

        const bool firstControlBit  = (firstControl >> laneIdx) & 1;
        const bool secondControlBit = (secondControl >> laneIdx) & 1;

        unsigned diff            = !firstControlBit & secondControlBit;
        const unsigned dataBytes = firstControlBit ? (secondControlBit ? 4 : 1) : 0;

        const unsigned dataBytesScan  = inclusiveScanInt(dataBytes);
        const unsigned dataBytesIndex = dataBytesScan - dataBytes;
        const unsigned warpDataBytes  = shflSync(dataBytesScan, GpuConfig::warpSize - 1);

        previous_ = shflSync(previous_, GpuConfig::warpSize - 1);

        for (unsigned i = 0; i < dataBytes; ++i)
            diff |= static_cast<unsigned>(vleData[dataBytesIndex + i]) << (8 * i);

        previous_ += inclusiveScanInt(diff + 1);

        buffer_ += 2 * sizeof(GpuConfig::ThreadMask) + warpDataBytes;
#else
        const std::uint8_t* vleData = buffer_ + sizeof(GpuConfig::ThreadMask);

        const auto readDataNibble = [vleData](unsigned index)
        {
            const unsigned byte = vleData[index / 2];
            return (byte >> ((index % 2) * 4)) & 0xf;
        };

        GpuConfig::ThreadMask nonOneBits = 0;
        for (unsigned i = 0; i < sizeof(GpuConfig::ThreadMask); ++i)
            nonOneBits |= GpuConfig::ThreadMask(buffer_[i]) << (8 * i);

        const bool nonOne = (nonOneBits >> laneIdx) & 1;

        const unsigned nNibbleIndex = popCount(nonOneBits & lanemask_lt());
        unsigned vleDataSize        = popCount(nonOneBits);

        const unsigned nNibblesData  = nonOne ? readDataNibble(nNibbleIndex) : 0;
        const bool additionalStorage = nonOne ? nNibblesData <= 7 : 0;
        const unsigned nNibbles      = additionalStorage ? nNibblesData + 1 : 0;

        const unsigned nbValueScan      = inclusiveScanInt(nNibbles);
        const unsigned nbValueDataIndex = vleDataSize + nbValueScan - nNibbles;
        const unsigned nbValueSize      = shflSync(nbValueScan, GpuConfig::warpSize - 1);
        vleDataSize += nbValueSize;

        previous_ = shflSync(previous_, GpuConfig::warpSize - 1);

        unsigned diff = nonOne ? (additionalStorage ? readDataNibble(nbValueDataIndex) : nNibblesData - 6) : 1;
        for (unsigned i = 1; i < nNibbles; ++i)
            diff |= readDataNibble(nbValueDataIndex + i) << (4 * i);

        previous_ += inclusiveScanInt(diff);

        buffer_ += sizeof(GpuConfig::ThreadMask) + (vleDataSize + 1) / 2;
#endif
        return previous_;
    }

private:
    const std::uint8_t* buffer_;
    unsigned numBytes_, numNeighbors_, previous_;
};

/*! compress a list of neighbor indices with a single warp
 *
 * This function compresses an array of neighbor indices using either the compression scheme proposed in 'Compressed
 * Neighbour Lists for SPH', by S. Band, C. Gissler and M. Teschner, 2020 or a custom nibble-based scheme, depending on
 * the CSTONE_USE_BAND_ET_AL_COMPRESSION macro.
 *
 * Note that the input values need to be the same for all threads in a warp. Caution: if the output buffer is too small,
 * it will overflow.
 *
 * @param[in]  neighbors  pointer to the array of neighbor indices to compress
 * @param[out] output     pointer to the output buffer where compressed data will be written
 * @param[in]  n          number of neighbor indices in the input array
 */
__device__ __forceinline__ unsigned
warpCompressNeighbors(const std::uint32_t* __restrict__ neighbors, void* __restrict__ output, const unsigned n)
{
    const unsigned laneIdx = laneIndex();
    WarpCompression compression(output);
    for (unsigned offset = 0; offset < n; offset += GpuConfig::warpSize)
    {
        const unsigned nb          = offset + laneIdx;
        const unsigned neighbor    = nb < n ? neighbors[nb] : 0;
        const unsigned activeLanes = std::min(n - offset, static_cast<unsigned>(GpuConfig::warpSize));
        compression.add(neighbor, activeLanes);
    }
    return compression.numBytes();
}

/*! decompress a list of neighbor indices which was compressed using warpCompressNeighbors with a single warp
 *
 * The function reads the compressed neighbor list from the input buffer and reconstructs
 * the original neighbor indices, storing them in the provided neighbors array.
 * The number of decompressed neighbor indices is returned via the reference parameter n.
 *
 * @param[in]  input     pointer to the buffer containing the compressed neighbor list
 * @param[out] neighbors pointer to the array where decompressed neighbor indices will be stored
 * @param[in]  n         expected number of neighbor indices (must match the number passed to warpCompressNeighbors)
 */
__device__ __forceinline__ void warpDecompressNeighbors(const void* const __restrict__ input,
                                                        std::uint32_t* const __restrict__ neighbors,
                                                        const unsigned n)
{
    const unsigned laneIdx = laneIndex();
    WarpDecompression decompression(input, n);
    for (unsigned offset = 0; offset < n; offset += GpuConfig::warpSize)
    {
        const unsigned nb       = offset + laneIdx;
        const unsigned neighbor = decompression.next();
        if (nb < n) neighbors[nb] = neighbor;
    }
}

} // namespace cstone
