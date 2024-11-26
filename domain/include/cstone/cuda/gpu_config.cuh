/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
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