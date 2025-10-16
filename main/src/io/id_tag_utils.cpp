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
#include <numeric>

#include "id_tag_utils.hpp"

namespace sphexa
{

uint64_t applyTaggingMask(uint64_t groupId, uint64_t id)
{
    if (groupId >= maxNumGroupIds)
        throw std::runtime_error("Tagging group id larger than max value (" + std::to_string(maxNumGroupIds) + ")\n");

    // Clear previous tagging bits if any
    uint64_t taggedId = id & ~taggingCheckMask;

    taggedId |= ((groupId + 1) << taggingMaskStartingBit);

    return taggedId;
}

void tagIdsInList(std::span<uint64_t> ids, std::size_t firstIndex, std::size_t lastIndex,
                  std::span<const uint64_t> selectedIds, std::span<const unsigned> selectedIdsGroups)
{
    if (selectedIds.size() != selectedIdsGroups.size())
        throw std::runtime_error("Number of selected ids and number of group ids must be the same\n");

#pragma omp parallel for schedule(static)
    for (auto i = firstIndex; i < lastIndex; i++)
    {
        // Since ids may be already tagged we need to unmask them in the search
        // Warning: race conditions can be generated here if selectedIds contains duplicates
        auto it = std::find(selectedIds.begin(), selectedIds.end(), ids[i] & ~taggingCheckMask);
        if (it != selectedIds.end())
        {
            auto index = it - selectedIds.begin();
            ids[i]    = applyTaggingMask(selectedIdsGroups[index], ids[i]);
        }
    }
}

void tagIdsInSphere(std::span<uint64_t> ids, std::span<const CoordinateType> x, std::span<const CoordinateType> y,
                    std::span<const CoordinateType> z, std::size_t firstIndex, std::size_t lastIndex,
                    std::span<const IdSelectionSphere> selSphereData, std::span<const unsigned> sphereGroupIds)
{

    if (selSphereData.size() != sphereGroupIds.size())
        throw std::runtime_error("Number of spherical volumes and number of group ids must be the same\n");\

    uint64_t groupIndex = 0;
    for (const auto& sphere : selSphereData)
    {
        const auto squareRadius = sphere[3] * sphere[3];
        const auto sphereCenter = util::makeVec3(sphere);
        const unsigned groupId = sphereGroupIds[groupIndex];
#pragma omp parallel for schedule(static)
        for (auto particleIndex = firstIndex; particleIndex < lastIndex; particleIndex++)
        {
            cstone::Vec3<CoordinateType> currentPosition{x[particleIndex], y[particleIndex], z[particleIndex]};
            auto                         squaredDistance = util::norm2(currentPosition - sphereCenter);
            if (squaredDistance < squareRadius) { ids[particleIndex] = applyTaggingMask(groupId, ids[particleIndex]); }
        }
        groupIndex++;
    }

}

template<class LocalIndexP>
void findTaggedIds(std::span<const uint64_t> ids, std::size_t firstIndex, std::size_t lastIndex,
                   std::vector<LocalIndexP>& taggedIdsIndexes)
{
    const auto               numIds = lastIndex - firstIndex;
    std::vector<uint8_t>     flags(numIds, 0);
    std::vector<LocalIndexP> flagsScan(numIds);

#pragma omp parallel for schedule(static)
    for (LocalIndexP index = 0; index < numIds; ++index)
    {
        flags[index] = IsMasked{}(ids[index + firstIndex]);
    }

    std::exclusive_scan(flags.begin(), flags.end(), flagsScan.begin(), uint32_t(0));
    taggedIdsIndexes.resize(flagsScan.back() + flags.back());

#pragma omp parallel for
    for (LocalIndexP i = 0; i < numIds; i++)
    {
        if (flags[i]) { taggedIdsIndexes[flagsScan[i]] = i + firstIndex; }
    }
}

template void findTaggedIds(std::span<const uint64_t> ids, std::size_t firstIndex, std::size_t lastIndex,
                            std::vector<uint32_t>& taggedIdsIndexes);
template void findTaggedIds(std::span<const uint64_t> ids, std::size_t firstIndex, std::size_t lastIndex,
                            std::vector<uint64_t>& taggedIdsIndexes);

} // namespace sphexa