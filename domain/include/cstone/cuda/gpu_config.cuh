/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  GPU hardware specific configuration
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cstdint>
#include "cstone/cuda/cuda_runtime.hpp"
#include "cstone/cuda/errorcheck.cuh"

namespace cstone
{

struct GpuConfig
{
//! @brief number of threads per warp
#if defined(__CUDACC__) && !defined(__HIPCC__)
    static constexpr int warpSize = 32;
#else
    static constexpr int warpSize = 64;
#endif

    static_assert(warpSize == 32 || warpSize == 64, "warp size has to be 32 or 64");

    //! @brief log2(warpSize)
    static constexpr int warpSizeLog2 = (warpSize == 32) ? 5 : 6;

    /*! @brief integer type for representing a thread mask, e.g. return value of __ballot_sync()
     *
     * This will automatically pick the right type based on the warpSize choice. Do not adapt.
     */
    using ThreadMask = std::conditional_t<warpSize == 32, uint32_t, uint64_t>;

    static int getSmCount()
    {
        cudaDeviceProp prop;
        checkGpuErrors(cudaGetDeviceProperties(&prop, 0));
        return prop.multiProcessorCount;
    }

    //! @brief number of multiprocessors
    inline static int smCount = getSmCount();
};

} // namespace cstone