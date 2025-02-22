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
//#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include "id_tag_utils.hpp"

namespace sphexa
{

/*! @brief Tagged id identification function  
 *
 * @param[in]  ids          ordered id list
 * @param[in]  first        first id index // TODO number of elements and pass iterator?
 * @param[in]  last         last (excluded) id index
 * @param[out] taggedIdsIndexes  vector of indexes (positions wrt of selected particles
 */
void findTaggedIds(const cstone::DeviceVector<uint64_t>& ids, size_t first, size_t last, std::vector<uint64_t>& taggedIdsIndexes)
{
    auto devRawId = ids.data();
    const auto devIdSize = last - first;

    // Create device containers
    thrust::device_vector<uint64_t> devMask(devIdSize, 0);
    thrust::device_vector<uint64_t> devScanResult(devIdSize, 0);

    // Generate mask
    thrust::transform(devRawId + first, devRawId + last, devMask.begin(), MaskFunctor{});

    // Run scan
    thrust::inclusive_scan(devMask.begin(), devMask.end(), devScanResult.begin());

    // Create particle subset position container on GPU and initialize it sequentially
    thrust::device_vector<uint64_t> devSubsetPos(devScanResult.back());
    thrust::sequence(thrust::device, devSubsetPos.begin(), devSubsetPos.end());
 
    // Find the position of the particle in the subset
    // TODO: can I use a zip iterator here instead of raw pointer?
    auto* devRawScanResult = thrust::raw_pointer_cast(devScanResult.data());
    const auto scanResultSize = devScanResult.size();
    SearchFunctor searchFunctor{devRawScanResult, scanResultSize};
    thrust::for_each(thrust::device, devSubsetPos.begin(), devSubsetPos.end(), searchFunctor);

    // Copy result to host
    // TODO: find better solution
    thrust::host_vector<uint64_t> hostSubsetPos(devSubsetPos);
    taggedIdsIndexes.assign(thrust::raw_pointer_cast(hostSubsetPos.data()), thrust::raw_pointer_cast(hostSubsetPos.data()) + hostSubsetPos.size());

    return;
}

}