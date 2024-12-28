/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Thrust allocator adaptor to prevent value initialization
 *
 * Taken from: https://github.com/NVIDIA/thrust/blob/master/examples/uninitialized_vector.cu
 */

#pragma once

#include <thrust/device_allocator.h>
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <cassert>

namespace util
{
// uninitialized_allocator is an allocator which
// derives from device_allocator and which has a
// no-op construct member function
template<typename T>
struct uninitialized_allocator : thrust::device_allocator<T>
{
    // the default generated constructors and destructors are implicitly
    // marked __host__ __device__, but the current Thrust device_allocator
    // can only be constructed and destroyed on the host; therefore, we
    // define these as host only
    __host__ uninitialized_allocator() {}
    __host__ uninitialized_allocator(const uninitialized_allocator& other)
        : thrust::device_allocator<T>(other)
    {
    }
    __host__ ~uninitialized_allocator() {}

    uninitialized_allocator& operator=(const uninitialized_allocator&) = default;

    // for correctness, you should also redefine rebind when you inherit
    // from an allocator type; this way, if the allocator is rebound somewhere,
    // it's going to be rebound to the correct type - and not to its base
    // type for U
    template<typename U>
    struct rebind
    {
        typedef uninitialized_allocator<U> other;
    };

    // note that construct is annotated as
    // a __host__ __device__ function
    __host__ __device__ void construct(T*)
    {
        // no-op
    }
};

} // namespace util
