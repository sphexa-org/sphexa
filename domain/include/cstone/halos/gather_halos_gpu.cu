
/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
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
 * @brief  Utility for GPU-direct halo particle exchange
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/cuda/cuda_runtime.hpp"
#include "cstone/primitives/math.hpp"
#include "cstone/primitives/stl.hpp"
#include "cstone/util/array.hpp"
#include "gather_halos_gpu.h"

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

template<class T, class IndexType>
__global__ void scatterRangesKernel(const IndexType* rangeScan,
                                    const IndexType* rangeOffsets,
                                    int numRanges,
                                    T* dest,
                                    const T* buffer,
                                    size_t bufferSize)
{
    IndexType tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < bufferSize)
    {
        IndexType rangeIdx = stl::upper_bound(rangeScan, rangeScan + numRanges, tid) - rangeScan - 1;

        IndexType destIdx = rangeOffsets[rangeIdx] + tid - rangeScan[rangeIdx];
        dest[destIdx]     = buffer[tid];
    }
}

template<class T, class IndexType>
void scatterRanges(const IndexType* rangeScan,
                   const IndexType* rangeOffsets,
                   int numRanges,
                   T* dest,
                   const T* buffer,
                   size_t bufferSize)
{
    int numThreads = 256;
    int numBlocks  = iceil(bufferSize, numThreads);
    scatterRangesKernel<<<numBlocks, numThreads>>>(rangeScan, rangeOffsets, numRanges, dest, buffer, bufferSize);
}

template void scatterRanges(const unsigned*, const unsigned*, int, uint8_t*, const uint8_t*, size_t);
template void scatterRanges(const unsigned*, const unsigned*, int, int*, const int*, size_t);
template void scatterRanges(const uint64_t*, const uint64_t*, int, int*, const int*, size_t);

template void
scatterRanges(const unsigned*, const unsigned*, int, util::array<float, 1>*, const util::array<float, 1>*, size_t);
template void
scatterRanges(const unsigned*, const unsigned*, int, util::array<float, 2>*, const util::array<float, 2>*, size_t);
template void
scatterRanges(const unsigned*, const unsigned*, int, util::array<float, 3>*, const util::array<float, 3>*, size_t);
template void
scatterRanges(const unsigned*, const unsigned*, int, util::array<float, 4>*, const util::array<float, 4>*, size_t);

} // namespace cstone
