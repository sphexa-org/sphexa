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

#include <cassert>
#include <cstdint>

#include "cstone/cuda/gpu_config.cuh"
#include "cstone/primitives/clz.hpp"
#include "cstone/primitives/warpscan.cuh"

namespace cstone
{

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
__device__ __forceinline__ void
warpCompressNeighbors(const std::uint32_t* __restrict__ neighbors, char* __restrict__ output, const unsigned n)
{
    const unsigned laneIdx = laneIndex();

    if (n == 0)
    {
        if (laneIdx == 0) *((unsigned*)output) = sizeof(unsigned);
        return;
    }

    std::uint8_t* buffer = (std::uint8_t*)((unsigned*)output + 1);
    unsigned previous    = unsigned(-1);

#if CSTONE_USE_BAND_ET_AL_COMPRESSION
    for (unsigned offset = 0; offset < n; offset += GpuConfig::warpSize)
    {
        std::uint8_t* vleData = buffer + 2 * sizeof(GpuConfig::ThreadMask);

        const unsigned nb           = offset + laneIdx;
        const unsigned neighbor     = nb < n ? neighbors[nb] : 0;
        const unsigned leftNeighbor = shflUpSync(neighbor, 1);
        const unsigned diff         = nb < n ? (neighbor - (laneIdx > 0 ? leftNeighbor : previous)) - 1 : 0;
        previous                    = shflSync(neighbor, GpuConfig::warpSize - 1);

        const auto firstControl   = ballotSync(diff > 1);
        const auto secondControl  = ballotSync((diff == 1) | (diff >= 256));
        const auto controlToStore = laneIdx < sizeof(GpuConfig::ThreadMask) ? firstControl : secondControl;
        if (laneIdx < 2 * sizeof(GpuConfig::ThreadMask))
            buffer[laneIdx] = controlToStore >> (8 * (laneIdx % sizeof(GpuConfig::ThreadMask)));

        const unsigned dataBytes      = diff >= 2 ? (diff >= 256 ? 4 : 1) : 0;
        const unsigned dataBytesScan  = inclusiveScanInt(dataBytes);
        const unsigned dataBytesIndex = dataBytesScan - dataBytes;
        const unsigned warpDataBytes  = shflSync(dataBytesScan, GpuConfig::warpSize - 1);

        for (unsigned i = 0; i < dataBytes; ++i)
            vleData[dataBytesIndex + i] = (diff >> (8 * i)) & 0xff;

        buffer += 2 * sizeof(GpuConfig::ThreadMask) + warpDataBytes;
    }
#else
    for (unsigned offset = 0; offset < n; offset += GpuConfig::warpSize)
    {
        std::uint8_t* vleData = buffer + sizeof(GpuConfig::ThreadMask);

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

        const unsigned nb           = offset + laneIdx;
        const unsigned neighbor     = nb < n ? neighbors[nb] : 0;
        const unsigned leftNeighbor = shflUpSync(neighbor, 1);
        const unsigned diff         = neighbor - (laneIdx > 0 ? leftNeighbor : previous);
        previous                    = shflSync(neighbor, GpuConfig::warpSize - 1);

        const bool nonOne     = diff != 1 & nb < n;
        const auto nonOneBits = ballotSync(nonOne);
        if (laneIdx < sizeof(GpuConfig::ThreadMask)) buffer[laneIdx] = (nonOneBits >> (8 * laneIdx));
        const bool additionalStorage = (diff > 9) & (nb < n);
        const unsigned nBits         = 32 - countLeadingZeros(diff);
        const unsigned nNibbles      = additionalStorage ? (nBits + 3) / 4 : 0;
        const unsigned nNibblesData  = additionalStorage ? nNibbles - 1 : diff + 6;

        const unsigned nNibblesIndex     = exclusiveScanBool(nonOne);
        unsigned vleDataSize             = 0;
        const unsigned nNibblesDataIndex = vleDataSize + nNibblesIndex;
        vleDataSize += popCount(nonOneBits);

        if (nonOne) writeDataNibble(nNibblesDataIndex, nNibblesData, false);
#ifdef __HIP_PLATFORM_AMD__
        // This should not be necessary, a memory fence should be enough, but tests fail without
        __syncthreads();
#else
        syncWarp();
#endif
        if (nonOne) writeDataNibble(nNibblesDataIndex, nNibblesData, true);

        const unsigned nbValueScan      = inclusiveScanInt(nNibbles);
        const unsigned nbValueDataIndex = vleDataSize + nbValueScan - nNibbles;
        const unsigned nbValueSize      = shflSync(nbValueScan, GpuConfig::warpSize - 1);
        vleDataSize += nbValueSize;

        for (unsigned i = 0; i < nNibbles; ++i)
            writeDataNibble(nbValueDataIndex + i, (diff >> (4 * i)) & 0xf, false);
#ifdef __HIP_PLATFORM_AMD__
        // This should not be necessary, a memory fence should be enough, but tests fail without
        __syncthreads();
#else
        syncWarp();
#endif
        for (unsigned i = 0; i < nNibbles; ++i)
            writeDataNibble(nbValueDataIndex + i, (diff >> (4 * i)) & 0xf, true);

        buffer += sizeof(GpuConfig::ThreadMask) + (vleDataSize + 1) / 2;
    }
#endif

    const unsigned totalBytes = (unsigned)(buffer - (std::uint8_t*)output);
    assert(totalBytes < (1 << 16));
    assert(n < (1 << 16));
    if (laneIdx == 0) *((unsigned*)output) = totalBytes | (n << 16);
}

/*! extract the size of a neighbor list compressed by warpCompressNeighbors
 *
 * @param[in] input pointer to the buffer containing the compressed neighbor list
 * @return          the size (in bytes) of the compressed neighbor list
 */
__device__ __forceinline__ unsigned compressedNeighborsSize(const char* const input)
{
    return *((const unsigned*)input) & 0xffff;
}

/*! decompress a list of neighbor indices which was compressed using warpCompressNeighbors with a single warp
 *
 * The function reads the compressed neighbor list from the input buffer and reconstructs
 * the original neighbor indices, storing them in the provided neighbors array.
 * The number of decompressed neighbor indices is returned via the reference parameter n.
 *
 * @param[in]  input     pointer to the buffer containing the compressed neighbor list
 * @param[out] neighbors pointer to the array where decompressed neighbor indices will be stored
 * @param[out] n         reference to an unsigned integer where the number of neighbor indices will be stored
 */
__device__ __forceinline__ void
warpDecompressNeighbors(const char* const __restrict__ input, std::uint32_t* const __restrict__ neighbors, unsigned& n)
{
    const unsigned laneIdx = laneIndex();

    n = *((unsigned*)input) >> 16;

    if (n == 0) return;

    const std::uint8_t* buffer = (const std::uint8_t*)((const unsigned*)input + 1);
    unsigned previous          = unsigned(-1);

#if CSTONE_USE_BAND_ET_AL_COMPRESSION
    for (unsigned offset = 0; offset < n; offset += GpuConfig::warpSize)
    {
        const std::uint8_t* vleData = buffer + 2 * sizeof(GpuConfig::ThreadMask);

        const unsigned nb = offset + laneIdx;

        GpuConfig::ThreadMask firstControl  = 0;
        GpuConfig::ThreadMask secondControl = 0;
        for (unsigned i = 0; i < sizeof(GpuConfig::ThreadMask); ++i)
        {
            firstControl |= GpuConfig::ThreadMask(buffer[i]) << (8 * i);
            secondControl |= GpuConfig::ThreadMask(buffer[sizeof(GpuConfig::ThreadMask) + i]) << (8 * i);
        }

        const bool firstControlBit  = (firstControl >> laneIdx) & 1;
        const bool secondControlBit = (secondControl >> laneIdx) & 1;

        unsigned diff            = !firstControlBit & secondControlBit;
        const unsigned dataBytes = firstControlBit ? (secondControlBit ? 4 : 1) : 0;

        const unsigned dataBytesScan  = inclusiveScanInt(dataBytes);
        const unsigned dataBytesIndex = dataBytesScan - dataBytes;
        const unsigned warpDataBytes  = shflSync(dataBytesScan, GpuConfig::warpSize - 1);

        previous = shflSync(previous, GpuConfig::warpSize - 1);

        for (unsigned i = 0; i < dataBytes; ++i)
            diff |= unsigned(vleData[dataBytesIndex + i]) << (8 * i);

        previous += inclusiveScanInt(diff + 1);

        if (nb < n) neighbors[nb] = previous;

        buffer += 2 * sizeof(GpuConfig::ThreadMask) + warpDataBytes;
    }
#else
    for (unsigned offset = 0; offset < n; offset += GpuConfig::warpSize)
    {
        const std::uint8_t* vleData = buffer + sizeof(GpuConfig::ThreadMask);

        const auto readDataNibble = [vleData](unsigned index)
        {
            const unsigned byte = vleData[index / 2];
            return (byte >> ((index % 2) * 4)) & 0xf;
        };

        GpuConfig::ThreadMask nonOneBits = 0;
        for (unsigned i = 0; i < sizeof(GpuConfig::ThreadMask); ++i)
            nonOneBits |= GpuConfig::ThreadMask(buffer[i]) << (8 * i);

        const bool nonOne = (nonOneBits >> laneIdx) & 1;

        unsigned vleDataSize        = 0;
        const unsigned nNibbleIndex = vleDataSize + popCount(nonOneBits & lanemask_lt());
        vleDataSize += popCount(nonOneBits);

        const unsigned nNibblesData  = nonOne ? readDataNibble(nNibbleIndex) : 0;
        const bool additionalStorage = nonOne ? nNibblesData <= 7 : 0;
        const unsigned nNibbles      = additionalStorage ? nNibblesData + 1 : 0;

        const unsigned nbValueScan      = inclusiveScanInt(nNibbles);
        const unsigned nbValueDataIndex = vleDataSize + nbValueScan - nNibbles;
        const unsigned nbValueSize      = shflSync(nbValueScan, GpuConfig::warpSize - 1);
        vleDataSize += nbValueSize;

        previous = shflSync(previous, GpuConfig::warpSize - 1);

        unsigned diff = nonOne ? (additionalStorage ? readDataNibble(nbValueDataIndex) : nNibblesData - 6) : 1;
        for (unsigned i = 1; i < nNibbles; ++i)
            diff |= readDataNibble(nbValueDataIndex + i) << (4 * i);

        previous += inclusiveScanInt(diff);
        const unsigned nb = offset + laneIdx;
        if (nb < n) neighbors[nb] = previous;

        buffer += sizeof(GpuConfig::ThreadMask) + (vleDataSize + 1) / 2;
    }
#endif
}

} // namespace cstone
