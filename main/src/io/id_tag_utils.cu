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

#include "id_tag_utils.hpp"

namespace sphexa
{

/*! @brief Id tagging (in first:last range) from list
 *
 * @param[out] ids          ordered id list
 * @param[in]  first        first id index // TODO number of elements and pass iterator?
 * @param[in]  last         last (excluded) id index
 * @param[in]  selectedIds  vindexes to be tagged
 */
void tagIdsInList(cstone::DeviceVector<IdType>& ids, size_t first, size_t last, const std::vector<IdType>& selectedIds) 
{
    throw std::runtime_error("Not implemented yet");
}


/*! @brief Id tagging (in first:last range) in spherical volume, GPU version
 *
 * @param[out] ids               ordered id list
 * @param[in]  x                 x coordinates
 * @param[in]  y                 y coordinates
 * @param[in]  z                 z coordinates
 * @param[in]  first             first id index // TODO number of elements and pass iterator?
 * @param[in]  last              last (excluded) id index
 * @param[in]  selSphereData     spherical volume definition
 */
void tagIdsInSphere(cstone::DeviceVector<IdType>& ids, const std::vector<CoordinateType>& x, const std::vector<CoordinateType>& y,
    const std::vector<CoordinateType>& z, size_t firstIndex, size_t lastIndex, const IdSelectionSphere& selSphereData)
{
    throw std::runtime_error("Not implemented yet");
}

/*! @brief Tagged id identification
 *
 * @param[in]  ids              ordered id list
 * @param[in]  first            first id index // TODO number of elements and pass iterator?
 * @param[in]  last             last (excluded) id index
 * @param[out] taggedIdsIndexes vector of indexes (positions wrt of selected particles)
 */
void findTaggedIds(const cstone::DeviceVector<IdType>& ids, size_t first, size_t last, std::vector<IdType>& taggedIdsIndexes)
{
    const auto devIdSize = last - first;
    thrust::device_vector<IdType> devMask(devIdSize);
    thrust::device_vector<IdType> devScanResult(devIdSize);

    // Generate mask
    thrust::transform(ids.data() + first, ids.data() + last, devMask.begin(), MaskFunctor{});

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
        // TODO: find better solution
        thrust::host_vector<IdType> hostSubsetPos(devSubsetPos);
        taggedIdsIndexes.assign(thrust::raw_pointer_cast(hostSubsetPos.data()), thrust::raw_pointer_cast(hostSubsetPos.data()) + hostSubsetPos.size());
    }
    else
    {
        taggedIdsIndexes.clear();
    }

    return;
}

}