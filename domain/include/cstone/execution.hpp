/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Portable CUDA/HIP stream type for use in CPU and GPU code
 *
 * This header provides only the cudaStream_t type without including the full
 * CUDA/HIP runtime headers, making it safe to include from .hpp headers
 * that are compiled by the host compiler.
 */

#pragma once

#include <type_traits>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)

#define cudaStream_t hipStream_t

typedef struct ihipStream_t* hipStream_t;
typedef hipStream_t cudaStream_t;

#else

// Forward declare CUDA stream type without including cuda_runtime.h
// to keep CPU translation units free of CUDA includes.
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;

#endif

namespace cstone::execution
{

struct Cpu
{
    constexpr static Cpu Default() { return {}; }
};

struct Gpu
{
    cudaStream_t stream;

    constexpr Gpu(cudaStream_t s)
        : stream(s)
    {
    }

    constexpr static Gpu Default() { return {0}; }

    operator cudaStream_t() const { return stream; }
};

template<class Execution>
using HaveGpu = std::is_same<Execution, Gpu>;

} // namespace cstone::execution
