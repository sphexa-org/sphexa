/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Warp-level primitives
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <type_traits>

#include "cstone/cuda/gpu_config.cuh"
#include "cstone/primitives/clz.hpp"

namespace cstone
{

__device__ __forceinline__ unsigned laneIndex()
{
#ifdef __CUDACC__
    unsigned laneIdx;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(laneIdx));
    return laneIdx;
#else
    return (threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x) & (GpuConfig::warpSize - 1);
#endif
}

//! @brief there's no int overload for min in AMD ROCM
__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ unsigned imin(unsigned a, unsigned b) { return a < b ? a : b; }

__device__ __forceinline__ uint32_t reverseBits(uint32_t x) { return __brev(x); }

__device__ __forceinline__ uint64_t reverseBits(uint64_t x) { return __brevll(x); }

template<class T, std::enable_if_t<sizeof(T) == 4 && std::is_integral_v<T>, int> = 0>
__device__ __forceinline__ int popCount(T x)
{
    return __popc(x);
}

template<class T, std::enable_if_t<sizeof(T) == 8 && std::is_integral_v<T>, int> = 0>
__device__ __forceinline__ int popCount(T x)
{
    return __popcll(x);
}

__device__ __forceinline__ void syncWarp()
{
#if defined(__CUDACC__)
    __syncwarp();
#endif
}

namespace detail
{

template<class T, class = void>
struct IsHardwareShuffleable : std::false_type
{
};

template<class T>
struct IsHardwareShuffleable<T,
                             std::void_t<decltype(
#if defined(__CUDACC__) && !defined(__HIPCC__)
                                 __shfl_sync(0xFFFFFFFF, std::declval<T>(), 0, 0)
#else
                                 __shfl(std::declval<T>(), 0)
#endif
                                     )>> : std::true_type
{
};

static_assert(IsHardwareShuffleable<int>::value);
static_assert(IsHardwareShuffleable<unsigned>::value);
static_assert(IsHardwareShuffleable<long>::value);
static_assert(IsHardwareShuffleable<unsigned long>::value);
static_assert(IsHardwareShuffleable<long long>::value);
static_assert(IsHardwareShuffleable<unsigned long long>::value);
static_assert(IsHardwareShuffleable<float>::value);
static_assert(IsHardwareShuffleable<double>::value);

template<class T, class Op>
__device__ __forceinline__ T shflSyncImpl(T value, Op&& shflOp)
{
    if constexpr (detail::IsHardwareShuffleable<T>::value) { return shflOp(value); }
    else
    {
        static_assert(std::is_trivially_copyable_v<T>);
        constexpr int numInts = (sizeof(T) + sizeof(int) - 1) / sizeof(int);
        int buffer[numInts];
        memcpy(buffer, &value, sizeof(value));
#pragma unroll
        for (int i = 0; i < numInts; ++i)
            buffer[i] = shflOp(buffer[i]);
        memcpy(&value, buffer, sizeof(value));
        return value;
    }
}

} // namespace detail

//! @brief Compatibility wrapper for AMD.
template<class T>
__device__ __forceinline__ T shflSync(T value, int srcLane)
{
    return detail::shflSyncImpl(value,
                                [=](auto v)
                                {
#if defined(__CUDACC__) && !defined(__HIPCC__)
                                    return __shfl_sync(0xFFFFFFFF, v, srcLane);
#else
                                    return __shfl(v, srcLane);
#endif
                                });
}

//! @brief Compatibility wrapper for AMD.
template<class T>
__device__ __forceinline__ T shflXorSync(T value, int laneMask)
{
    return detail::shflSyncImpl(value,
                                [=](auto v)
                                {
#if defined(__CUDACC__) && !defined(__HIPCC__)
                                    return __shfl_xor_sync(0xFFFFFFFF, v, laneMask);
#else
                                    return __shfl_xor(v, laneMask);
#endif
                                });
}

//! @brief Compatibility wrapper for AMD.
template<class T>
__device__ __forceinline__ T shflUpSync(T value, int distance)
{
    return detail::shflSyncImpl(value,
                                [=](auto v)
                                {
#if defined(__CUDACC__) && !defined(__HIPCC__)
                                    return __shfl_up_sync(0xFFFFFFFF, v, distance);
#else
                                    return __shfl_up(v, distance);
#endif
                                });
}

//! @brief Compatibility wrapper for AMD.
template<class T>
__device__ __forceinline__ T shflDownSync(T value, int distance)
{
    return detail::shflSyncImpl(value,
                                [=](auto v)
                                {
#if defined(__CUDACC__) && !defined(__HIPCC__)
                                    return __shfl_down_sync(0xFFFFFFFF, v, distance);
#else
                                    return __shfl_down(v, distance);
#endif
                                });
}

//! @brief Compatibility wrapper for AMD.
__device__ __forceinline__ GpuConfig::ThreadMask ballotSync(bool flag)
{
#if defined(__CUDACC__) && !defined(__HIPCC__)
    return __ballot_sync(0xFFFFFFFF, flag);
#else
    return __ballot(flag);
#endif
}

//! @brief Compatibility wrapper for AMD.
__device__ __forceinline__ GpuConfig::ThreadMask anySync(bool flag)
{
#if defined(__CUDACC__) && !defined(__HIPCC__)
    return __any_sync(0xFFFFFFFF, flag);
#else
    return __any(flag);
#endif
}

//! @brief compute warp-wide min
template<class T>
__device__ __forceinline__ T warpMin(T laneVal)
{
#pragma unroll
    for (int i = 0; i < GpuConfig::warpSizeLog2; i++)
    {
        laneVal = min(laneVal, shflXorSync(laneVal, 1 << i));
    }

    return laneVal;
}

//! @brief compute warp-wide max
template<class T>
__device__ __forceinline__ T warpMax(T laneVal)
{
#pragma unroll
    for (int i = 0; i < GpuConfig::warpSizeLog2; i++)
    {
        laneVal = max(laneVal, shflXorSync(laneVal, 1 << i));
    }

    return laneVal;
}

//! @brief compute warp-wide bitwise or
template<class T>
__device__ __forceinline__ T warpBitwiseOr(T laneVal)
{
#pragma unroll
    for (int i = 0; i < GpuConfig::warpSizeLog2; i++)
    {
        laneVal |= shflXorSync(laneVal, 1 << i);
    }

    return laneVal;
}

//! @brief standard inclusive warp-scan
__device__ __forceinline__ int inclusiveScanInt(int value)
{
    unsigned lane = laneIndex();
#pragma unroll
    for (int i = 1; i < GpuConfig::warpSize; i *= 2)
    {
        int partial = shflUpSync(value, i);
        if (i <= lane) { value += partial; }
    }
    return value;
}

//! @brief returns a mask with bits set for each warp lane before the calling lane
__device__ __forceinline__ GpuConfig::ThreadMask lanemask_lt()
{
    GpuConfig::ThreadMask lane = laneIndex();
    return (GpuConfig::ThreadMask(1) << lane) - 1;
}

__device__ __forceinline__ int exclusiveScanBool(const bool p)
{
    using SignedMask        = std::make_signed_t<GpuConfig::ThreadMask>;
    GpuConfig::ThreadMask b = ballotSync(p);
    return popCount(SignedMask(b & lanemask_lt()));
}

__device__ __forceinline__ int reduceBool(const bool p)
{
    using SignedMask        = std::make_signed_t<GpuConfig::ThreadMask>;
    GpuConfig::ThreadMask b = ballotSync(p);
    return popCount(SignedMask(b));
}

// Segmented scan int

//! @brief returns a mask with bits set for each warp lane before and including the calling lane
__device__ __forceinline__ GpuConfig::ThreadMask lanemask_le()
{
    GpuConfig::ThreadMask lane = laneIndex();
    return (GpuConfig::ThreadMask(2) << lane) - 1;
}

/*! @brief perform range-limited inclusive warp scan
 *
 * @param value     scan input per lane
 * @param distance  number of preceding lanes to include in scanned result
 *                  distance has to be <= the current laneIdx
 * @return          the scanned value
 *
 * Due to the @p distance argument, the values of preceding lanes are only added
 * to the current lane if the source lane is less than @p distance away from the current lane.
 * If distance == current laneIdx, then the result is the same as a normal inclusive scan.
 */
__device__ __forceinline__ int inclusiveSegscan(int value, int distance)
{
#pragma unroll
    for (int i = 1; i < GpuConfig::warpSize; i *= 2)
    {
        int partial = shflUpSync(value, i);
        if (i <= distance) { value += partial; }
    }

    return value;
}

/*! @brief performs an inclusive segmented warp prefix scan
 *
 * @param[in] packedValue input value per lane
 * @param[in] carryValue  offset for first prefix sum segment
 * @return                the prefix sum value per lane
 *
 * If packedValue is positive, behavior is like a usual prefix sum
 * Each negative packedValue will start a new prefix sum segment with a value of -packedValue - 1
 */
__device__ __forceinline__ int inclusiveSegscanInt(const int packedValue, const int carryValue)
{
    int laneIdx = laneIndex();

    int isNegative = packedValue < 0;
    int mask       = -isNegative;

    // value = packedValue if packedValue >= 0
    // value = -packedValue - 1 if packedValue < 0
    int value = (~mask & packedValue) + (mask & (-1 - packedValue));

    GpuConfig::ThreadMask flags = ballotSync(isNegative);

    // distance = number of preceding lanes to include in scanned value
    // e.g. if distance = 0, then no preceding lane value will be added to scannedValue
    int distance     = countLeadingZeros(flags & lanemask_le()) + laneIdx - (GpuConfig::warpSize - 1);
    int scannedValue = inclusiveSegscan(value, imin(distance, laneIdx));

    // the lowest lane index for which packedValue was negative, warpSize if all were positive
    int firstNegativeLane = countLeadingZeros(reverseBits(flags));
    int addCarry          = -(laneIdx < firstNegativeLane);

    return scannedValue + (carryValue & addCarry);
}

/*! @brief warp-level stream compaction
 *
 * @tparam       T            an elementary type
 * @param[inout] value        input value per lane
 * @param[in]    keep         keep value of current lane yes/no
 * @param[-]     sm_exchange  temporary work space, uninitialized
 * @return                    number of keep flags with a value of true within the warp
 *
 * Discards values of lanes for which keep=false and move the valid values to the first numKeep lanes.
 * Example:
 *  lane   0  1  2  3  4  5  6  7
 *  value  0 10 20 30 40 50 60 70
 *  keep   1  0  1  0  1  0  1  0
 *  -----------------------------
 *  return:
 *  value  0 20 40 60  0  0  0  0
 */
template<class T>
__device__ __forceinline__ int streamCompact(T* value, bool keep, volatile T* sm_exchange)
{
    using SignedMask                 = std::make_signed_t<GpuConfig::ThreadMask>;
    GpuConfig::ThreadMask keepBallot = ballotSync(keep);
    // exclusive scan of keepBallot
    int laneCompacted = popCount(SignedMask(keepBallot & lanemask_lt()));

    if (keep) { sm_exchange[laneCompacted] = *value; }
    syncWarp();

    int laneIdx = laneIndex();
    *value      = sm_exchange[laneIdx];

    int numKeep = popCount(SignedMask(keepBallot));
    return numKeep;
}

/*! @brief spread first warpSize/8 lanes and perform segmented scan
 *
 * @param val 10 20 30 40
 * @return    10 11 12 13 14 15 16 17 20 21 22 23 24 25 26 27 30 31 32 33 34 35 36 37 40 41 42 43 44 45 46 47
 *
 */
__device__ __forceinline__ int spreadSeg8(int val)
{
    int laneIdx = laneIndex();
    return shflSync(val, laneIdx >> 3) + (laneIdx & 7);
}

// source: https://stackoverflow.com/a/72461459 CC BY-SA 4.0 by user timothygiraffe
__device__ __forceinline__ float atomicMinFloat(float* addr, float value)
{
    return !signbit(value) ? __int_as_float(atomicMin((int*)addr, __float_as_int(value)))
                           : __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));
}

// source: https://stackoverflow.com/a/72461459 CC BY-SA 4.0 by user timothygiraffe
__device__ __forceinline__ float atomicMaxFloat(float* addr, float value)
{
    return !signbit(value) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
                           : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
}

} // namespace cstone
