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

__device__ __forceinline__ void
warpCompressNeighbors(const std::uint32_t* __restrict__ neighbors, char* __restrict__ output, const unsigned n)
{
    // TODO: add a buffer size limit, currently we just overflow

    const unsigned laneIdx = laneIndex();

    if (n == 0)
    {
        if (laneIdx == 0) *((unsigned*)output) = 0;
        return;
    }

#if CSTONE_USE_BAND_ET_AL_COMPRESSION
    GpuConfig::ThreadMask* control = (GpuConfig::ThreadMask*)output + 1;
    std::uint32_t* first = (std::uint32_t*)(control + (n - 1 + GpuConfig::warpSize - 1) / GpuConfig::warpSize * 2);
    std::uint8_t* data   = (std::uint8_t*)(first + 1);

    unsigned dataSize = 0;
    unsigned previous = neighbors[0];
    if (laneIdx == 0) *first = previous;
    for (unsigned offset = 1; offset < n; offset += GpuConfig::warpSize)
    {
        const unsigned nb           = offset + laneIdx;
        const unsigned neighbor     = nb < n ? neighbors[nb] : 0;
        const unsigned leftNeighbor = shflUpSync(neighbor, 1);
        const unsigned diff         = nb < n ? (neighbor - (laneIdx > 0 ? leftNeighbor : previous)) - 1 : 0;
        previous                    = shflSync(neighbor, GpuConfig::warpSize - 1);

        const auto firstControl  = ballotSync(diff > 1);
        const auto secondControl = ballotSync((diff == 1) | (diff >= 256));
        if (laneIdx == 0)
        {
            control[2 * ((offset - 1) / GpuConfig::warpSize)]     = firstControl;
            control[2 * ((offset - 1) / GpuConfig::warpSize) + 1] = secondControl;
        }

        const unsigned dataBytes      = diff >= 2 ? (diff >= 256 ? 4 : 1) : 0;
        const unsigned dataBytesScan  = inclusiveScanInt(dataBytes);
        const unsigned dataBytesIndex = dataSize + dataBytesScan - dataBytes;
        dataSize += shflSync(dataBytesScan, GpuConfig::warpSize - 1);

        for (unsigned i = 0; i < dataBytes; ++i)
            data[dataBytesIndex + i] = (diff >> (8 * i)) & 0xff;
    }

    const unsigned totalBytes =
        sizeof(GpuConfig::ThreadMask) * (1 + (n - 1 + GpuConfig::warpSize - 1) / GpuConfig::warpSize * 2) + 4 +
        dataSize;
    assert(n < (1 << 16));
    if (laneIdx == 0) *((unsigned*)output) = totalBytes | (n << 16);
#else
    GpuConfig::ThreadMask* nonOnes = (GpuConfig::ThreadMask*)output + 1;
    std::uint8_t* data             = (std::uint8_t*)(nonOnes + (n + GpuConfig::warpSize - 1) / GpuConfig::warpSize);

    const auto writeDataNibble = [&](unsigned index, std::uint8_t value, bool odd)
    {
        assert(value < 16);
        if (odd == index % 2)
        {
            std::uint8_t byte = odd ? data[index / 2] : 0;
            byte |= (value << ((index % 2) * 4));
            data[index / 2] = byte;
        }
    };

    unsigned dataSize = 0;
    unsigned previous = 0;
    for (unsigned offset = 0; offset < n; offset += GpuConfig::warpSize)
    {
        const unsigned nb           = offset + laneIdx;
        const unsigned neighbor     = nb < n ? neighbors[nb] : 0;
        const unsigned leftNeighbor = shflUpSync(neighbor, 1);
        const unsigned diff         = neighbor - (laneIdx > 0 ? leftNeighbor : previous);
        previous                    = shflSync(neighbor, GpuConfig::warpSize - 1);

        const bool nonOne     = diff != 1 & nb < n;
        const auto nonOneBits = ballotSync(nonOne);
        if (laneIdx == 0) nonOnes[offset / GpuConfig::warpSize] = nonOneBits;
        const bool additionalStorage = (diff < 1 | diff >= 9) & (nb < n);
        const unsigned nBits         = diff == 0 ? 1 : 32 - countLeadingZeros(diff);
        const unsigned nNibbles      = additionalStorage ? (nBits + 3) / 4 : 0;
        const unsigned nNibblesData  = additionalStorage ? nNibbles : diff + 7;

        const unsigned nNibblesIndex     = exclusiveScanBool(nonOne);
        const unsigned nNibblesDataIndex = dataSize + nNibblesIndex;
        const unsigned nNibblesSize      = shflSync(nNibblesIndex + nonOne, GpuConfig::warpSize - 1);
        dataSize += nNibblesSize;

        if (nonOne) writeDataNibble(nNibblesDataIndex, nNibblesData, false);
#ifdef __HIP_PLATFORM_AMD__
        // This should not be necessary, a memory fence should be enough, but tests fail without
        __syncthreads();
#else
        syncWarp();
#endif
        if (nonOne) writeDataNibble(nNibblesDataIndex, nNibblesData, true);

        const unsigned nbValueScan      = inclusiveScanInt(nNibbles);
        const unsigned nbValueDataIndex = dataSize + nbValueScan - nNibbles;
        const unsigned nbValueSize      = shflSync(nbValueScan, GpuConfig::warpSize - 1);
        dataSize += nbValueSize;

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
    }

    const unsigned totalBytes =
        sizeof(GpuConfig::ThreadMask) * (1 + (n + GpuConfig::warpSize - 1) / GpuConfig::warpSize) + (dataSize + 1) / 2;
    assert(n < (1 << 16));
    if (laneIdx == 0) *((unsigned*)output) = totalBytes | (n << 16);
#endif
}

__device__ __forceinline__ unsigned compressedNeighborsSize(const char* const input)
{
    return *((const unsigned*)input) & 0xffff;
}

__device__ __forceinline__ void
warpDecompressNeighbors(const char* const __restrict__ input, std::uint32_t* const __restrict__ neighbors, unsigned& n)
{
    const unsigned laneIdx = laneIndex();

    n = *((unsigned*)input) >> 16;

    if (n == 0) return;

#if CSTONE_USE_BAND_ET_AL_COMPRESSION
    const GpuConfig::ThreadMask* control = (const GpuConfig::ThreadMask*)input + 1;
    const std::uint32_t* first =
        (std::uint32_t*)(control + (n - 1 + GpuConfig::warpSize - 1) / GpuConfig::warpSize * 2);
    const std::uint8_t* data = (std::uint8_t*)(first + 1);

    unsigned dataSize = 0;
    unsigned previous = *first;
    if (laneIdx == 0) neighbors[0] = previous;
    for (unsigned offset = 1; offset < n; offset += GpuConfig::warpSize)
    {
        const unsigned nb        = offset + laneIdx;
        const auto firstControl  = control[2 * ((offset - 1) / GpuConfig::warpSize)];
        const auto secondControl = control[2 * ((offset - 1) / GpuConfig::warpSize) + 1];

        const bool firstControlBit  = (firstControl >> laneIdx) & 1;
        const bool secondControlBit = (secondControl >> laneIdx) & 1;

        unsigned diff            = !firstControlBit & secondControlBit;
        const unsigned dataBytes = firstControlBit ? (secondControlBit ? 4 : 1) : 0;

        const unsigned dataBytesScan  = inclusiveScanInt(dataBytes);
        const unsigned dataBytesIndex = dataSize + dataBytesScan - dataBytes;
        dataSize += shflSync(dataBytesScan, GpuConfig::warpSize - 1);

        previous = shflSync(previous, GpuConfig::warpSize - 1);

        for (unsigned i = 0; i < dataBytes; ++i)
            diff |= unsigned(data[dataBytesIndex + i]) << (8 * i);

        previous += inclusiveScanInt(diff + 1);

        if (nb < n) neighbors[nb] = previous;
    }
#else
    const GpuConfig::ThreadMask* nonOnes = (const GpuConfig::ThreadMask*)input + 1;
    const std::uint8_t* data = (const std::uint8_t*)(nonOnes + (n + GpuConfig::warpSize - 1) / GpuConfig::warpSize);

    const auto readDataNibble = [data](unsigned index)
    {
        const unsigned byte = data[index / 2];
        return (byte >> ((index % 2) * 4)) & 0xf;
    };

    unsigned dataSize = 0;
    unsigned previous = 0;
    for (unsigned offset = 0; offset < n; offset += GpuConfig::warpSize)
    {
        const auto nonOneBits = nonOnes[offset / GpuConfig::warpSize];
        const unsigned nb     = offset + laneIdx;
        const bool nonOne     = (nonOneBits >> laneIdx) & 1;

        const unsigned nNibblesIndex     = exclusiveScanBool(nonOne);
        const unsigned nNibblesDataIndex = dataSize + nNibblesIndex;
        const unsigned nNibblesSize      = shflSync(nNibblesIndex + nonOne, GpuConfig::warpSize - 1);
        dataSize += nNibblesSize;

        const unsigned nNibblesData  = nonOne ? readDataNibble(nNibblesDataIndex) : 0;
        const bool additionalStorage = nonOne ? nNibblesData <= 8 : 0;
        const unsigned nNibbles      = additionalStorage ? nNibblesData : 0;

        const unsigned nbValueScan      = inclusiveScanInt(nNibbles);
        const unsigned nbValueDataIndex = dataSize + nbValueScan - nNibbles;
        const unsigned nbValueSize      = shflSync(nbValueScan, GpuConfig::warpSize - 1);
        dataSize += nbValueSize;

        previous = shflSync(previous, GpuConfig::warpSize - 1);

        unsigned diff = nonOne ? (additionalStorage ? readDataNibble(nbValueDataIndex) : nNibblesData - 7) : 1;
        for (unsigned i = 1; i < nNibbles; ++i)
            diff |= readDataNibble(nbValueDataIndex + i) << (4 * i);

        previous += inclusiveScanInt(diff);
        if (nb < n) neighbors[nb] = previous;
    }
#endif
}

} // namespace cstone
