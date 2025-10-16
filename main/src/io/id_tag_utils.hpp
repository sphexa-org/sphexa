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

#include <span>
#include <vector>

#include "cstone/tree/definitions.h"
#include "sph/types.hpp"

namespace sphexa
{

using CoordinateType = sph::SphTypes::CoordinateType;

/*! @brief Number of bits used for tagging information storage
 */
constexpr uint64_t tagNumBits = 10;

/*! @brief Given tagNumBits, the maximum number of groups we can address is 2^tagNumBits - 1.
 * We subtract one, because groupId=0 corresponds to an unmasked particle ID
 */
constexpr uint64_t maxNumGroupIds = (1 << tagNumBits) - 1;

/*! @brief First tagging bit position
 */
constexpr uint64_t taggingMaskStartingBit = sizeof(uint64_t) * 8 - tagNumBits;

constexpr uint64_t taggingCheckMask = maxNumGroupIds << taggingMaskStartingBit;

/*! @brief Tagged id identification condition functor
 */
struct IsMasked
{
    HOST_DEVICE_FUN uint8_t operator()(uint64_t id) const { return (id & taggingCheckMask) != 0; }
};

/*! @brief Application of tagging mask to a given id
 *
 * @param[in]  groupId      selection group id (must be < maxNumGroupIds)
 * @param[in] id            input id
 * @return                  tagged id
 */
uint64_t applyTaggingMask(uint64_t groupId, uint64_t id);

/*! @brief Tagged id (in first:last range) identification, CPU version
 *
 * @param[in]  ids               ordered id list
 * @param[in]  firstIndex        first id index
 * @param[in]  lastIndex         last (excluded) id index
 * @param[out] taggedIdsIndexes  vector of indexes of tagged ids
 */
template<class LocalIndexP>
extern void findTaggedIds(std::span<const uint64_t> ids, std::size_t firstIndex, std::size_t lastIndex,
                          std::vector<LocalIndexP>& taggedIdsIndexes);

/*! @brief Tagged id (in first:last range) identification, GPU version
 *
 * @param[in]  ids               ordered id list
 * @param[in]  firstIndex        first id index
 * @param[in]  lastIndex         last (excluded) id index
 * @param[out] taggedIdsIndexes  vector with indices of tagged ids
 */
template<class LocalIndexP, template<class> class DeviceVector>
extern void findTaggedIdsGPU(std::span<const uint64_t> ids, std::size_t firstIndex, std::size_t lastIndex,
                             DeviceVector<LocalIndexP>& taggedIdsIndexes);

/*! @brief Id tagging spherical volume definition
 * (center[0:2], radius)
 */
using IdSelectionSphere = cstone::Vec4<CoordinateType>;

/*! @brief Id tagging (in first:last range) from list
 *
 * @param[out] ids               id list
 * @param[in]  firstIndex        first id index
 * @param[in]  lastIndex         last (excluded) id index
 * @param[in]  selectedIdsList   ids to be tagged (no duplications allowed)
 * @param[in]  selectedIdsGroups group id for each selected id
 */
void tagIdsInList(std::span<uint64_t> ids, std::size_t firstIndex, std::size_t lastIndex,
                  std::span<const uint64_t> selectedIds, std::span<const unsigned> selectedIdsGroups);


/*! @brief Id tagging (in first:last range) in spherical volume
 *
 * @param[out] ids               id list
 * @param[in]  x                 x coordinates
 * @param[in]  y                 y coordinates
 * @param[in]  z                 z coordinates
 * @param[in]  firstIndex        first id index
 * @param[in]  lastIndex         last (excluded) id index
 * @param[in]  selSphereData     set of spherical volume definitions
 * @param[in]  sphereGroupIds    group id for each spherical volume definition
 */
void tagIdsInSphere(std::span<uint64_t> ids, std::span<const CoordinateType> x, std::span<const CoordinateType> y,
                    std::span<const CoordinateType> z, std::size_t firstIndex, std::size_t lastIndex,
                    std::span<const IdSelectionSphere> selSphereData, std::span<const unsigned> sphereGroupIds);
} // namespace sphexa