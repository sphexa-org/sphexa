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
 * @brief  CPU/GPU Particle ID tag utilities
 *
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <numeric>
#include <span>
#include <vector>

#include "cstone/tree/definitions.h"
#include "sph/types.hpp"

namespace sphexa
{

using IdType = uint64_t;
using LocalIndex = uint32_t;
using CoordinateType = sph::SphTypes::CoordinateType;

/*! @brief Number of bits used for tagging information storage
 */
constexpr IdType taggingMaskSize = 10;

/*! @brief Given the taggingMaskSize, this is the maximum selection group id value plus one
 */
constexpr IdType supGroupId = (1 << taggingMaskSize) - 1;

/*! @brief First tagging bit position
 */
constexpr IdType taggingMaskStartingBit = sizeof(IdType)*8 - taggingMaskSize;

constexpr IdType taggingCheckMask = supGroupId << taggingMaskStartingBit;


/*! @brief Tagged id identification condition functor
 */
struct IsMasked
{
    HOST_DEVICE_FUN uint8_t operator()(IdType id) const
    {
        return (id & taggingCheckMask) != 0;
    }
};

/*! @brief Application of tagging mask to a given id
 *
 * @param[in]  groupId      selection group id (must be < supGroupId)
 * @param[out] id           input id with tagging mask applied
 */
void applyTaggingMask(IdType groupId, IdType& id);

/*! @brief Tagged id (in first:last range) identification, CPU version
 *
 * @param[in]  ids          ordered id list
 * @param[in]  first        first id index // TODO number of elements and pass iterator?
 * @param[in]  last         last (excluded) id index
 * @param[out] taggedIdsIndexes  vector of indexes of tagged ids
 */
template<class IdTypeP, class LocalIndexP>
void findTaggedIds(std::span<const IdTypeP> ids, size_t first, size_t last, std::vector<LocalIndexP>& taggedIdsIndexes);
extern template void findTaggedIds<IdType, LocalIndex>(std::span<const IdType> ids, size_t first, size_t last, std::vector<LocalIndex>& taggedIdsIndexes);

/*! @brief Tagged id (in first:last range) identification, GPU version
 *
 * @param[in]  ids          ordered id list
 * @param[in]  first        first id index // TODO number of elements and pass iterator?
 * @param[in]  last         last (excluded) id index
 * @param[out] taggedIdsIndexes  vector of indexes of tagged ids
 */
template<class IdTypeP, class LocalIndexP>
void findTaggedIdsGPU(std::span<const IdTypeP> ids, size_t first, size_t last, std::vector<LocalIndexP>& taggedIdsIndexes);
extern template void findTaggedIdsGPU<IdType, LocalIndex>(std::span<const IdType> ids, size_t first, size_t last, std::vector<LocalIndex>& taggedIdsIndexes);


/*! @brief Id tagging spherical volume definition
 * (center[0:2], radius)
 */
using IdSelectionSphere = cstone::Vec4<CoordinateType>;

/*! @brief Id tagging list definition
 */
using IdSelectionList = std::vector<IdType>;

/*! @brief Id tagging (in first:last range) from list, CPU version
 *
 * @param[out] ids               id list
 * @param[in]  first             first id index // TODO number of elements and pass iterator?
 * @param[in]  last              last (excluded) id index
 * @param[in]  selectedIds       indexes to be tagged
 * @param[in]  groupId           selection group id
 */
template<class IdTypeP>
void tagIdsInList(std::span<IdTypeP> ids, size_t first, size_t last, std::span<const IdTypeP> selectedIds, const LocalIndex groupId = 0);
extern template void tagIdsInList<IdType>(std::span<IdType> ids, size_t first, size_t last, std::span<const IdType> selectedIds,
    const LocalIndex groupId);

/*! @brief Id tagging (in first:last range) in spherical volume, CPU version
 *
 * @param[out] ids               id list
 * @param[in]  x                 x coordinates
 * @param[in]  y                 y coordinates
 * @param[in]  z                 z coordinates
 * @param[in]  first             first id index // TODO number of elements and pass iterator?
 * @param[in]  last              last (excluded) id index
 * @param[in]  selSphereData     spherical volume definition
 * @param[in]  groupId           selection group id
 */
template<class IdTypeP>
void tagIdsInSphere(std::span<IdTypeP> ids, std::span<const CoordinateType> x, std::span<const CoordinateType> y,
    std::span<const CoordinateType> z, size_t firstIndex, size_t lastIndex, IdSelectionSphere selSphereData,
    const LocalIndex groupId = 0);
extern template void tagIdsInSphere<IdType>(std::span<IdType> ids, std::span<const CoordinateType> x, std::span<const CoordinateType> y,
    std::span<const CoordinateType> z, size_t firstIndex, size_t lastIndex, IdSelectionSphere selSphereData,
    const LocalIndex groupId);
}