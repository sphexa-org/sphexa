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
#include <iostream>

#include "id_tag_utils.hpp"

namespace sphexa
{

void applyTaggingMask(IdType groupId, IdType& id)
{
    if (groupId >= supGroupId)
        throw std::runtime_error("Tagging group id larger than max value (" + std::to_string(supGroupId) + ")\n");

    // Clear previous tagging if present
    if(IsMasked{}(id)) {
        id &= ~taggingCheckMask;
        std::cout << "Warning: applying tagging mask to already tagged id " << id << "\n";
    }

    // Apply new tagging information
    id |= ((groupId+1) << taggingMaskStartingBit);
}

template<class IdTypeP>
void tagIdsInList(std::span<IdTypeP> ids, size_t first, size_t last, std::span<const IdTypeP> selectedIds,
    const LocalIndex groupId)
{
    const auto idListBeginIt = ids.begin()+first;
    const auto idListEndIt = ids.begin()+last;
    auto lastFound = 0;
    std::for_each(selectedIds.begin(), selectedIds.end(), [idListBeginIt, idListEndIt, &lastFound, groupId](auto selectedIds){
        auto lower = std::lower_bound(idListBeginIt+lastFound, idListEndIt, selectedIds);
        if(lower != idListEndIt && *lower == selectedIds) {
            lastFound = lower - idListBeginIt + 1;
            applyTaggingMask(groupId, *lower);
        }
    });
}
template void tagIdsInList<IdType>(std::span<IdType> ids, size_t first, size_t last, std::span<const IdType> selectedIds,
    const LocalIndex groupId);


template<class IdTypeP>
void tagIdsInSphere(std::span<IdTypeP> ids, std::span<const CoordinateType> x, std::span<const CoordinateType> y,
    std::span<const CoordinateType> z, size_t firstIndex, size_t lastIndex, IdSelectionSphere selSphereData,
    const LocalIndex groupId)
{
    const auto squareRadius = selSphereData[3]*selSphereData[3];
    const auto sphereCenter = util::makeVec3(selSphereData);
#pragma omp parallel for schedule(static)
    for(auto particleIndex = firstIndex; particleIndex < lastIndex; particleIndex++){
        cstone::Vec3<CoordinateType> currentPosition{x[particleIndex], y[particleIndex], z[particleIndex]};
        auto squaredDistance = util::norm2(currentPosition - sphereCenter);
        if(squaredDistance < squareRadius) {
            applyTaggingMask(groupId, ids[particleIndex]);
        }
    }
}
template void tagIdsInSphere<IdType>(std::span<IdType> ids, std::span<const CoordinateType> x, std::span<const CoordinateType> y,
    std::span<const CoordinateType> z, size_t firstIndex, size_t lastIndex, IdSelectionSphere selSphereData,
    const LocalIndex groupId);


template<class IdTypeP, class LocalIndexP>
void findTaggedIds(std::span<const IdTypeP> ids, size_t first, size_t last, std::vector<LocalIndexP>& taggedIdsIndexes)
{
    const auto numIds = last - first;
    std::vector<uint8_t> flags(numIds, 0);
    std::vector<LocalIndexP> flagsScan(numIds);

#pragma omp parallel for schedule(static)
    for (LocalIndexP index = 0; index < numIds; ++index)
    {
        flags[index] = IsMasked{}(ids[index + first]);
    }

    std::exclusive_scan(flags.begin(), flags.end(), flagsScan.begin(), LocalIndex(0));
    taggedIdsIndexes.resize(flagsScan.back() + flags.back());

#pragma omp parallel for
    for (LocalIndexP i = 0; i < numIds; i++)
    {
        if (flags[i]) { taggedIdsIndexes[flagsScan[i]] = i + first; }
    }
}

template void findTaggedIds<IdType, LocalIndex>(std::span<const IdType> ids, size_t first, size_t last, std::vector<LocalIndex>& taggedIdsIndexes);

}