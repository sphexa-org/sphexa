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
 * @brief  CPU/GPU Particle ID tag utilities, CPU implementations
 *
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <omp.h>

#include <algorithm>
#include <execution>

#include "id_tag_utils.hpp"

namespace sphexa
{

/*! @brief Id tagging (in first:last range) from list, CPU version
 *
 * @param[out] ids          ordered id list
 * @param[in]  first        first id index // TODO number of elements and pass iterator?
 * @param[in]  last         last (excluded) id index
 * @param[in]  selectedIds  indexes to be tagged
 */
void tagIdsInList(IdVectorType& ids, size_t first, size_t last, const IdVectorType& selectedIds)
{
    const auto idListBeginIt = ids.begin()+first;
    const auto idListEndIt = ids.begin()+last;
    auto lastFound = 0;
    std::for_each(selectedIds.begin(), selectedIds.end(), [idListBeginIt, idListEndIt, &lastFound](auto selectedIds){
        auto lower = std::lower_bound(idListBeginIt+lastFound, idListEndIt, selectedIds);
        if(lower != idListEndIt && *lower == selectedIds) {
            lastFound = lower - idListBeginIt + 1;
            *lower = *lower | msbMask;
        }
    });
}

// TODO: should we save here the list of tagged ids without tag?
/*! @brief Id tagging (in first:last range) in spherical volume, CPU version
 *
 * @param[out] ids                ordered id list
 * @param[in]  x                  x coordinates
 * @param[in]  y                  y coordinates
 * @param[in]  z                  z coordinates
 * @param[in]  first              first id index // TODO number of elements and pass iterator?
 * @param[in]  last               last (excluded) id index
 * @param[in]  selSphereData      spherical volume definition
 */
void tagIdsInSphere(IdVectorType& ids, const std::vector<CoordinateType>& x, const std::vector<CoordinateType>& y,
    const std::vector<CoordinateType>& z, size_t firstIndex, size_t lastIndex, const IdSelectionSphere& selSphereData)
{
    const auto squareRadius = selSphereData.radius*selSphereData.radius;
#pragma omp parallel for schedule(static)
    for(auto particleIndex = firstIndex; particleIndex < lastIndex; particleIndex++){
        cstone::Vec3<CoordinateType> currentPosition{x[particleIndex], y[particleIndex], z[particleIndex]};
        auto squaredDistance = util::norm2(currentPosition - selSphereData.center);
        if(squaredDistance < squareRadius) {
            ids[particleIndex] = ids[particleIndex] | msbMask;
        }
    }
}

#if 0

// TODO: to be removed togheter with the findTaggedIds below
void exclusive_scan(const std::vector<IdType>& input, std::vector<IdType>& output)
{
    IdType n = input.size();
    unsigned int nthreads = 1;
    std::vector<IdType> thread_sums;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp single
        {
            nthreads = omp_get_num_threads();
            thread_sums.resize(nthreads + 1, 0);
        }

        IdType sum = 0;
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i)
        {
            output[i] = sum;
            sum += input[i];
        }
        thread_sums[tid + 1] = sum;

        #pragma omp barrier
        #pragma omp single
        {
            for (int i = 1; i <= nthreads; ++i)
                thread_sums[i] += thread_sums[i - 1];
        }

        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i)
        {
            output[i] += thread_sums[tid];
        }
    }
}


// TODO: to be removed, the fastest findTaggedIds implementation is the next one
/*! @brief Tagged id identification function
 *
 * @param[in]  ids          ordered id list
 * @param[in]  first        first id index // TODO number of elements and pass iterator?
 * @param[in]  last         last (excluded) id index
 * @param[out] taggedIdsIndexes  vector of indexes (positions wrt of selected particles
 */
void findTaggedIds(const IdVectorType& ids, size_t first, size_t last, IdVectorType& taggedIdsIndexes)
{
    const IdType hostIdSize = last - first;
    std::vector<IdType> hostMask(hostIdSize);
    std::vector<IdType> hostScanResult(hostIdSize);

    // Generate mask
//    std::transform(std::execution::par, ids.begin()+first, ids.begin()+last, hostMask.begin(), MaskFunctor{});
    #pragma omp parallel for
    for(IdType index=0; index<hostIdSize; ++index)
    {
        hostMask[index] = MaskFunctor{}(ids[index+first]);
    }


    // Run scan
//    std::exclusive_scan(std::execution::par, hostMask.begin(), hostMask.end(), hostScanResult.begin(), 0);
    exclusive_scan(hostMask, hostScanResult);

    if(hostScanResult.back() > 0 || hostMask.back() == 1) {

        // Scatter the tagged ids positions
        taggedIdsIndexes.resize(hostScanResult.back());
        if(hostMask.back() == 1) {
            taggedIdsIndexes.resize(hostScanResult.back()+1);
        }
        std::vector<IdType> hostSequence(hostIdSize);
        #pragma omp parallel for
        for(IdType i=0; i<hostIdSize; i++)
        {
            hostSequence[i] = first + i;
        }
//        std::iota(hostSequence.begin(), hostSequence.end(), first);

        #pragma omp parallel for
        for(IdType i=0; i<hostIdSize; i++)
        {
            if(hostMask[i])
            {
                taggedIdsIndexes[hostScanResult[i]] = hostSequence[i];
            }
        }
    }
    else {
        taggedIdsIndexes.clear();
    }

    return;
}
#else
/*! @brief Tagged id identification function  
 *
 * @param[in]  ids          ordered id list
 * @param[in]  first        first id index // TODO number of elements and pass iterator?
 * @param[in]  last         last (excluded) id index
 * @param[out] taggedIdsIndexes  vector of indexes (positions wrt of selected particles
 */
void findTaggedIds(const IdVectorType& ids, size_t first, size_t last, IdVectorType& taggedIdsIndexes)
{
    const IdType hostIdSize = last - first;
    taggedIdsIndexes.clear();
    taggedIdsIndexes.reserve(hostIdSize);

    #pragma omp parallel
    {
        IdVectorType tmpTaggedIdsIndexes;
//        tmpTaggedIdsIndexes.reserve(hostIdSize); // TODO: without a better estimate of the size, this is not efficient
        #pragma omp for nowait
        for (IdType index = first; index<last; ++index)
        {
            if (MaskFunctor{}(ids[index]))
                tmpTaggedIdsIndexes.push_back(index);
        }
        #pragma omp critical
        taggedIdsIndexes.insert(taggedIdsIndexes.end(), tmpTaggedIdsIndexes.begin(), tmpTaggedIdsIndexes.end());
    }
    std::sort(std::execution::par, taggedIdsIndexes.begin(), taggedIdsIndexes.end());
}
#endif

}