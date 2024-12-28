/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Defines macros for enabling device code compilation
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

// This will compile the annotated function as device AND host code in cuda translation units
// and as host functions in .cpp units
#if defined(__CUDACC__) || defined(__HIPCC__)
#define HOST_DEVICE_FUN __host__ __device__
#else
#define HOST_DEVICE_FUN
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define DEVICE_INLINE __forceinline__
#else
#define DEVICE_INLINE
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define HOST_DEVICE_INLINE __forceinline__
#else
#define HOST_DEVICE_INLINE inline
#endif
