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


//TODO: check headers
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>

#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

#include "id_tag_utils.hpp"

namespace sphexa
{

// TODO: the following two functions provides two different implementations of the tagged id identification.
// The first one uses a prefix scan + scatter based algorithm, while the second one just uses a copy_if step:
// according to some performace tests, namely the turbulence case of SPH-RUN with 1000^3
// particles and a synthetic data case with 1B ids and 10% tagged ids (see find_tagged_ids_test.cpp),
// the second one is ~2.6x/2.8x faster

#if 0
/*! @brief Tagged id identification
 *
 * @param[in]  ids              ordered id list
 * @param[in]  first            first id index // TODO number of elements and pass iterator?
 * @param[in]  last             last (excluded) id index
 * @param[out] taggedIdsIndexes vector of indexes (positions wrt of provided ids list)
 */
void findTaggedIdsGPU(std::span<const IdType> ids, size_t first, size_t last, std::vector<LocalIndex>& taggedIdsIndexes)
{
    const auto devIdSize = last - first;
    thrust::device_vector<IdType> devMask(devIdSize);
    thrust::device_vector<IdType> devScanResult(devIdSize);

    // Generate mask
    thrust::transform(ids.data() + first, ids.data() + last, devMask.begin(), IsMasked{});

    // Run scan
    thrust::exclusive_scan(devMask.begin(), devMask.end(), devScanResult.begin());

    if(devScanResult.back() > 0 || devMask.back() == 1)
    {
        // Scatter the tagged ids positions
        thrust::device_vector<IdType> devSubsetPos(devScanResult.back());
        if(devMask.back() == 1){
            devSubsetPos.resize(devScanResult.back()+1);
        }
        thrust::device_vector<IdType> devSequence(devIdSize);
        thrust::sequence(thrust::device, devSequence.begin(), devSequence.end(), first);
        thrust::scatter_if(thrust::device, devSequence.begin(), devSequence.end(), devScanResult.begin(), devMask.begin(), devSubsetPos.begin());

        // Copy result to host
        taggedIdsIndexes.resize(devSubsetPos.size());
        thrust::copy(devSubsetPos.begin(), devSubsetPos.end(), taggedIdsIndexes.begin());
    }
    else
    {
        taggedIdsIndexes.clear();
    }

    return;
}
#else

// TODO: move to hpp in case of unification with CPU version
struct IsMaskedGPU
{
    const IdType* ids_ptr;

    __host__ __device__
    IsMaskedGPU(const IdType* ids_ptr_) : ids_ptr(ids_ptr_) {}

    __device__
    bool operator()(LocalIndex idx) const
    {
        return IsMasked{}(ids_ptr[idx]);
    }
};

/*! @brief Tagged id identification
 *
 * @param[in]  ids              ordered id list
 * @param[in]  first            first id index // TODO number of elements and pass iterator?
 * @param[in]  last             last (excluded) id index
 * @param[out] taggedIdsIndexes vector of indexes (positions wrt of provided ids list)
 */
template<class IdTypeP, class LocalIndexP>
void findTaggedIdsGPU(std::span<const IdTypeP> ids, size_t first, size_t last, std::vector<LocalIndexP>& taggedIdsIndexes)
{

    // Count number of tagged ids
    IsMaskedGPU isMasked(ids.data());
    auto begin = thrust::make_counting_iterator<IdType>(first);
    auto end   = thrust::make_counting_iterator<IdType>(last);
    const size_t nTaggedIds = thrust::count_if(thrust::device, begin, end, isMasked);

    // Save indexes of tagged ids
    thrust::device_vector<IdType> deviceTaggedIdsIndexes(last - first);
    deviceTaggedIdsIndexes.resize(nTaggedIds);
    thrust::copy_if(thrust::device, begin, end, deviceTaggedIdsIndexes.begin(), isMasked);

    // Copy indices of tagged ids to host vector
    taggedIdsIndexes.clear();
    taggedIdsIndexes.resize(nTaggedIds);
    thrust::copy(deviceTaggedIdsIndexes.begin(), deviceTaggedIdsIndexes.end(), taggedIdsIndexes.begin());

    return;

}

template void findTaggedIdsGPU<IdType, LocalIndex>(std::span<const IdType> ids, size_t first, size_t last, std::vector<LocalIndex>& taggedIdsIndexes);

#endif
}