/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Utilities for Thrust device vectors for use in .cu translation units only
 */

#pragma once

#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>

#if defined(__HIPCC__)
#include <thrust/system/hip/execution_policy.h>
#else
#include <thrust/system/cuda/execution_policy.h>
#endif

#include "cuda_runtime.hpp"
#include "cstone/execution.hpp"

namespace cstone
{

template<class T, class Alloc>
T* rawPtr(thrust::device_vector<T, Alloc>& p)
{
    return thrust::raw_pointer_cast(p.data());
}

template<class T, class Alloc>
const T* rawPtr(const thrust::device_vector<T, Alloc>& p)
{
    return thrust::raw_pointer_cast(p.data());
}

template<class T, class Alloc>
T* rawPtr(thrust::universal_vector<T, Alloc>& p)
{
    return thrust::raw_pointer_cast(p.data());
}

template<class T, class Alloc>
const T* rawPtr(const thrust::universal_vector<T, Alloc>& p)
{
    return thrust::raw_pointer_cast(p.data());
}

inline auto thrustExecPolicy(execution::Gpu exec)
{
#if defined(__HIPCC__)
    return thrust::hip::par.on(exec);
#else
    return thrust::cuda::par.on(exec);
#endif
}

} // namespace cstone
