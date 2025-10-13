/*
 * MIT License
 *
 * SPH-EXA
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * @brief  CPU/GPU Particle ID tag utilities, GPU implementation
 *
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>

#include "id_tag_utils.hpp"

namespace sphexa
{

template<class LocalIndexP>
void findTaggedIdsGPU(std::span<const uint64_t> ids, std::size_t first, std::size_t last,
                      std::vector<LocalIndexP>& taggedIdsIndexes)
{

    const auto                     numIds = last - first;
    thrust::device_vector<uint8_t> flags(numIds, 0);

    IsMasked isMasked;
    thrust::transform(thrust::device, ids.data() + first, ids.data() + last, flags.begin(), isMasked);

    thrust::device_vector<LocalIndexP> flagsScan(numIds);
    thrust::exclusive_scan(flags.begin(), flags.end(), flagsScan.begin(), LocalIndexP(0), thrust::plus<LocalIndexP>());
    taggedIdsIndexes.resize(flagsScan.back() + flags.back());

    thrust::device_vector<LocalIndexP> taggedIdsIndexesDev(taggedIdsIndexes.size());
    thrust::scatter_if(thrust::device, thrust::make_counting_iterator(first),
                       thrust::make_counting_iterator(first + numIds), flagsScan.begin(), flags.begin(),
                       taggedIdsIndexesDev.begin());
    thrust::copy(taggedIdsIndexesDev.begin(), taggedIdsIndexesDev.end(), taggedIdsIndexes.begin());
}

template void findTaggedIdsGPU(std::span<const uint64_t> ids, std::size_t first, std::size_t last,
                               std::vector<uint32_t>& taggedIdsIndexes);
template void findTaggedIdsGPU(std::span<const uint64_t> ids, std::size_t first, std::size_t last,
                               std::vector<uint64_t>& taggedIdsIndexes);

} // namespace sphexa