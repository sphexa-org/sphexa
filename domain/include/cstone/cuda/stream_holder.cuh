/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief RAII GPU stream holder
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include "cstone/execution.hpp"
#include "cuda_runtime.hpp"
#include "errorcheck.cuh"

namespace cstone
{

class StreamHolder final
{
    cudaStream_t stream = 0;

public:
    StreamHolder() { checkGpuErrors(cudaStreamCreate(&stream)); }
    ~StreamHolder()
    {
        if (stream) { checkGpuErrors(cudaStreamDestroy(stream)); }
    }

    StreamHolder(const StreamHolder&) = delete;
    StreamHolder(StreamHolder&& other) { *this = std::move(other); }
    StreamHolder& operator=(const StreamHolder&) = delete;
    StreamHolder& operator=(StreamHolder&& other)
    {
        if (this != &other)
        {
            if (stream) checkGpuErrors(cudaStreamDestroy(stream));
            stream       = other.stream;
            other.stream = 0;
        }
        return *this;
    }

    operator cudaStream_t() const noexcept { return stream; }
    execution::Gpu exec() const noexcept { return execution::gpuStream(stream); }
    void sync() const { checkGpuErrors(cudaStreamSynchronize(stream)); }
};

} // namespace cstone
