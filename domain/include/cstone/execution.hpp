/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Execution policies for CPU and GPU execution
 */

#pragma once

#include <type_traits>

// Forward declare CUDA/HIP stream type without including cuda_runtime.h
// to keep CPU translation units free of CUDA/HIP includes.
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define cudaStream_t hipStream_t
typedef struct ihipStream_t* hipStream_t;
typedef hipStream_t cudaStream_t;
#else
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;
#endif

namespace cstone::execution
{

struct Cpu
{
};

struct Gpu
{
    operator cudaStream_t() const { return stream; }

private:
    constexpr explicit Gpu(cudaStream_t stream) noexcept
        : stream(stream)
    {
    }

    friend constexpr Gpu gpuStream(cudaStream_t stream) noexcept;

    cudaStream_t stream;
};

template<class Execution>
using HaveGpu = std::is_same<Execution, Gpu>;

constexpr inline Cpu cpu;

constexpr inline Gpu gpuStream(cudaStream_t stream) noexcept { return Gpu(stream); }

constexpr inline Gpu gpuDefaultStream = gpuStream(0);

namespace detail
{

template<class Exec>
struct DefaultExec
{
    static constexpr Exec value = {};
};

template<>
struct DefaultExec<Gpu>
{
    static constexpr Gpu value = gpuDefaultStream;
};

} // namespace detail

template<class T>
constexpr T defaultExec = detail::DefaultExec<T>::value;

} // namespace cstone::execution
