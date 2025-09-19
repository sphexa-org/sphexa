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

// TODO: the following two functions provides two different implementations of the tagged id identification.
// The first one uses a prefix scan + scatter based algorithm, similar to the corresponding GPU case
// while the second one just uses a copy_if step, again as in the corresponding GPU case.
// According to some performace tests, namely the turbulence case of SPH-RUN with 1000^3
// particles and a synthetic data case with 1B ids and 10% tagged ids (see find_tagged_ids_test.cpp),
// the second one is ~30x (synthetic data) faster and 0.1x (turbulence) slower than the first one with 64 threads.
// To put the above performance numbers into perspective, for 1000^3 particles and a 10% tagged ids, the identication
// in the slow case takes around 0.05s.
#if 1

/*! @brief Tagged id identification
 *
 * @param[in]  ids              ordered id list
 * @param[in]  first            first id index // TODO number of elements and pass iterator?
 * @param[in]  last             last (excluded) id index
 * @param[out] taggedIdsIndexes vector of indexes (positions wrt of provided ids list)
 */
/*
template<class IdType, class LocalIndex>
void findTaggedIds(std::span<const IdType> ids, LocalIndex first, LocalIndex last, std::vector<LocalIndex>& taggedIdsIndexes)
 */
void findTaggedIds(const IdVectorType& ids, size_t first, size_t last, IdVectorType& taggedIdsIndexes)
{
    using LocalIndex = uint32_t;
    const IdType numIds = last - first;
    std::vector<uint8_t> flags(numIds);
    std::vector<LocalIndex> flagsScan(numIds);

#pragma omp parallel for schedule(static)
    for (LocalIndex index = 0; index < numIds; ++index)
    {
        flags[index] = IsMasked{}(ids[index + first]);
    }

    std::exclusive_scan(flags.begin(), flags.end(), flagsScan.begin(), LocalIndex(0));
    taggedIdsIndexes.resize(flagsScan.back() + flags.back());

#pragma omp parallel for
    for (LocalIndex i = 0; i < numIds; i++)
    {
        if (flags[i]) { taggedIdsIndexes[flagsScan[i]] = i + first; }
    }
}

// template void findTaggedIds(std::span<const uint64_t> ids, uint32_t first, uint32_t last, std::vector<uint32_t>&
// taggedIdsIndexes));
#else
/*! @brief Tagged id identification
 *
 * @param[in]  ids              ordered id list
 * @param[in]  first            first id index // TODO number of elements and pass iterator?
 * @param[in]  last             last (excluded) id index
 * @param[out] taggedIdsIndexes vector of indexes (positions wrt of provided ids list)
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
            if (IsMasked{}(ids[index]))
                tmpTaggedIdsIndexes.push_back(index);
        }
        #pragma omp critical
        taggedIdsIndexes.insert(taggedIdsIndexes.end(), tmpTaggedIdsIndexes.begin(), tmpTaggedIdsIndexes.end());
    }
    std::sort(taggedIdsIndexes.begin(), taggedIdsIndexes.end());
}
#endif

}