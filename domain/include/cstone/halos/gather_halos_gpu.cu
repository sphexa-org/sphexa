/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Utility for GPU-direct halo particle exchange
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/cuda/cuda_runtime.hpp"
#include "cstone/primitives/math.hpp"
#include "cstone/primitives/stl.hpp"
#include "cstone/util/array.hpp"
#include "gather_halos_gpu.h"

#include <cstone/tree/definitions.h>

namespace cstone
{

template<class T, class IndexType>
__global__ void gatherRangesKernel(const IndexType* rangeScan,
                                   const IndexType* rangeOffsets,
                                   int numRanges,
                                   const T* src,
                                   T* buffer,
                                   size_t bufferSize)
{
    IndexType tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < bufferSize)
    {
        IndexType rangeIdx = stl::upper_bound(rangeScan, rangeScan + numRanges, tid) - rangeScan - 1;

        IndexType srcIdx = rangeOffsets[rangeIdx] + tid - rangeScan[rangeIdx];
        buffer[tid]      = src[srcIdx];
    }
}

template<class T, class IndexType>
void gatherRanges(const IndexType* rangeScan,
                  const IndexType* rangeOffsets,
                  int numRanges,
                  const T* src,
                  T* buffer,
                  size_t bufferSize)
{
    int numThreads = 256;
    int numBlocks  = iceil(bufferSize, numThreads);
    gatherRangesKernel<<<numBlocks, numThreads>>>(rangeScan, rangeOffsets, numRanges, src, buffer, bufferSize);
}

template void gatherRanges(const unsigned*, const unsigned*, int, const int*, int*, size_t);
template void gatherRanges(const uint64_t*, const uint64_t*, int, const int*, int*, size_t);

template void
gatherRanges(const unsigned*, const unsigned*, int, const util::array<float, 1>*, util::array<float, 1>*, size_t);
template void
gatherRanges(const unsigned*, const unsigned*, int, const util::array<float, 2>*, util::array<float, 2>*, size_t);
template void
gatherRanges(const unsigned*, const unsigned*, int, const util::array<float, 3>*, util::array<float, 3>*, size_t);
template void
gatherRanges(const unsigned*, const unsigned*, int, const util::array<float, 4>*, util::array<float, 4>*, size_t);

template void
gatherRanges(const uint64_t*, const uint64_t*, int, const util::array<float, 1>*, util::array<float, 1>*, size_t);
template void
gatherRanges(const uint64_t*, const uint64_t*, int, const util::array<float, 2>*, util::array<float, 2>*, size_t);
template void
gatherRanges(const uint64_t*, const uint64_t*, int, const util::array<float, 3>*, util::array<float, 3>*, size_t);
template void
gatherRanges(const uint64_t*, const uint64_t*, int, const util::array<float, 4>*, util::array<float, 4>*, size_t);

} // namespace cstone
