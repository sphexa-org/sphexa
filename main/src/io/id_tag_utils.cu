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

// TODO: to be removed, used to select between two implementations
#if 0

struct IsMaskedGPU
{
    const IdType* ids_ptr;

    IsMaskedGPU(const IdType* ids_ptr_) : ids_ptr(ids_ptr_) {}

    HOST_DEVICE_FUN
    uint8_t operator()(LocalIndex idx) const
    {
       return IsMasked{}(ids_ptr[idx]);
    }
};

// TODO: this implementation is 3x faster than the one below
template<class IdTypeP, class LocalIndexP>
void findTaggedIdsGPU(std::span<const IdTypeP> ids, size_t first, size_t last, std::vector<LocalIndexP>& taggedIdsIndexes)
{

    // Count number of tagged ids
    IsMaskedGPU isMasked(ids.data());
    auto begin = thrust::make_counting_iterator<IdType>(first);
    auto end   = thrust::make_counting_iterator<IdType>(last);
    const size_t nTaggedIds = thrust::count_if(thrust::device, begin, end, isMasked);

    // Save indexes of tagged ids
    thrust::device_vector<IdType> deviceTaggedIdsIndexes(nTaggedIds);
    thrust::copy_if(thrust::device, begin, end, deviceTaggedIdsIndexes.begin(), isMasked);

    // Copy indices of tagged ids to host vector
    taggedIdsIndexes.resize(nTaggedIds);
    thrust::copy(deviceTaggedIdsIndexes.begin(), deviceTaggedIdsIndexes.end(), taggedIdsIndexes.begin());

    return;

}

#else

template<class IdTypeP, class LocalIndexP>
void findTaggedIdsGPU(std::span<const IdTypeP> ids, size_t first, size_t last, std::vector<LocalIndexP>& taggedIdsIndexes)
{

    const auto numIds = last - first;
    thrust::device_vector<uint8_t> flags(numIds, 0);

    IsMasked isMasked;
    thrust::transform(thrust::device, ids.data() + first, ids.data() + last, flags.begin(), isMasked);

    thrust::device_vector<LocalIndexP> flagsScan(numIds);
    thrust::exclusive_scan(flags.begin(), flags.end(), flagsScan.begin(), 0, thrust::plus<LocalIndexP>());
    taggedIdsIndexes.resize(flagsScan.back() + flags.back());

    thrust::device_vector<LocalIndexP> taggedIdsIndexesDev(taggedIdsIndexes.size());
    thrust::scatter_if(thrust::device, thrust::make_counting_iterator(first), thrust::make_counting_iterator(first + numIds),
            flagsScan.begin(), flags.begin(), taggedIdsIndexesDev.begin());
    thrust::copy(taggedIdsIndexesDev.begin(), taggedIdsIndexesDev.end(), taggedIdsIndexes.begin());

}

#endif

template void findTaggedIdsGPU<IdType, LocalIndex>(std::span<const IdType> ids, size_t first, size_t last, std::vector<LocalIndex>& taggedIdsIndexes);

}