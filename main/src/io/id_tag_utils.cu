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

#include "cstone/cuda/device_vector.h"
#include "id_tag_utils.hpp"

namespace sphexa
{

struct FindAndTagIdsInList
{
    FindAndTagIdsInList(const thrust::device_vector<uint64_t>& selectedIds, const thrust::device_vector<unsigned>& selectedIdsGroups)
        : selectedIdsSize(selectedIds.size())
        , selectedIdsPtr(thrust::raw_pointer_cast(selectedIds.data()))
        , selectedIdsGroupsPtr(thrust::raw_pointer_cast(selectedIdsGroups.data()))
    {}

    HOST_DEVICE_FUN uint64_t operator()(uint64_t id) const
    {
        // Since ids may be already tagged we need to unmask them in the search
        uint64_t untaggedId = id & ~taggingCheckMask;
        for(std::size_t i = 0; i < selectedIdsSize; ++i)
        {
            if (selectedIdsPtr[i] == untaggedId)
            {
                return applyTaggingMask(selectedIdsGroupsPtr[i], id);
            }
        }
        return id;
    }

    uint64_t selectedIdsSize;
    const uint64_t* selectedIdsPtr;
    const unsigned* selectedIdsGroupsPtr;
};

struct FindAndTagInSphere
{
    FindAndTagInSphere(const thrust::device_vector<IdSelectionSphere>& selSpheres, const thrust::device_vector<unsigned>& sphereGroupIds)
        : numSpheres(selSpheres.size())
        , selSpheresPtr(thrust::raw_pointer_cast(selSpheres.data()))
        , sphereGroupIdsPtr(thrust::raw_pointer_cast(sphereGroupIds.data()))
    {}

    HOST_DEVICE_FUN uint64_t operator()(const thrust::tuple<uint64_t, CoordinateType, CoordinateType, CoordinateType>& particleData) const
    {
        uint64_t currentId = get<0>(particleData);
        const cstone::Vec3<CoordinateType> currentPosition{get<1>(particleData), get<2>(particleData), get<3>(particleData)};
        for (unsigned groupIndex = 0; groupIndex < numSpheres; ++groupIndex)
        {
            const auto     sphere          = selSpheresPtr[groupIndex];
            const auto     squareRadius    = sphere[3] * sphere[3];
            const auto     sphereCenter    = util::makeVec3(sphere);
            auto           squaredDistance = util::norm2(currentPosition - sphereCenter);
            if (squaredDistance < squareRadius)
            {
                currentId = applyTaggingMask(sphereGroupIdsPtr[groupIndex], currentId);
            }
        }
        return currentId;
    }

    uint64_t numSpheres;
    const IdSelectionSphere* selSpheresPtr;
    const unsigned* sphereGroupIdsPtr;
};

void tagIdsInListGPU(std::span<uint64_t> ids, std::size_t firstIndex, std::size_t lastIndex,
                     std::span<const uint64_t> selectedIds, std::span<const unsigned> selectedIdsGroups)
{
    if (selectedIds.size() != selectedIdsGroups.size())
        throw std::runtime_error("Number of selected ids and number of group ids must be the same\n");

    thrust::device_vector<uint64_t> selectedIdsDev(selectedIds.begin(), selectedIds.end());
    thrust::device_vector<unsigned> selectedIdsGroupsDev(selectedIdsGroups.begin(), selectedIdsGroups.end());
    FindAndTagIdsInList findAndTagIdsInList(selectedIdsDev, selectedIdsGroupsDev);
    thrust::transform(thrust::device, ids.data() + firstIndex, ids.data() + lastIndex, ids.data() + firstIndex, findAndTagIdsInList);
}

void tagIdsInSphereGPU(std::span<uint64_t> ids, std::span<const CoordinateType> x, std::span<const CoordinateType> y,
                       std::span<const CoordinateType> z, std::size_t firstIndex, std::size_t lastIndex,
                       std::span<const IdSelectionSphere> selSpheres, std::span<const unsigned> sphereGroupIds)
{
    if (selSpheres.size() != sphereGroupIds.size())
        throw std::runtime_error("Number of spherical volumes and number of group ids must be the same\n");

    auto first = thrust::make_zip_iterator(thrust::make_tuple(ids.data() + firstIndex, x.data() + firstIndex,
                                                        y.data() + firstIndex,  z.data() + firstIndex));
    auto last  = thrust::make_zip_iterator(thrust::make_tuple(ids.data() + lastIndex,  x.data() + lastIndex,
                                                        y.data() + lastIndex,  z.data() + lastIndex));
    thrust::device_vector<IdSelectionSphere> selSpheresDev(selSpheres.begin(), selSpheres.end());
    thrust::device_vector<unsigned> sphereGroupIdsDev(sphereGroupIds.begin(), sphereGroupIds.end());
    FindAndTagInSphere findAndTagInSphere(selSpheresDev, sphereGroupIdsDev);
    thrust::transform(thrust::device, first, last, ids.data() + firstIndex, findAndTagInSphere);
}

template<class LocalIndexP, template<class> class DeviceVector>
extern void findTaggedIdsGPU(std::span<const uint64_t> ids, std::size_t firstIndex, std::size_t lastIndex,
                             DeviceVector<LocalIndexP>& taggedIdsIndexes)
{

    const auto                     numIds = lastIndex - firstIndex;
    thrust::device_vector<uint8_t> flags(numIds, 0);

    IsMasked isMasked;
    thrust::transform(thrust::device, ids.data() + firstIndex, ids.data() + lastIndex, flags.begin(), isMasked);

    thrust::device_vector<LocalIndexP> flagsScan(numIds);
    thrust::exclusive_scan(flags.begin(), flags.end(), flagsScan.begin(), LocalIndexP(0), thrust::plus<LocalIndexP>());
    taggedIdsIndexes.resize(flagsScan.back() + flags.back());

    thrust::scatter_if(thrust::device, thrust::make_counting_iterator(firstIndex),
                       thrust::make_counting_iterator(firstIndex + numIds), flagsScan.begin(), flags.begin(),
                       taggedIdsIndexes.data());
}

template void findTaggedIdsGPU(std::span<const uint64_t> ids, std::size_t firstIndex, std::size_t lastIndex,
                               cstone::DeviceVector<uint32_t>& taggedIdsIndexes);
template void findTaggedIdsGPU(std::span<const uint64_t> ids, std::size_t firstIndex, std::size_t lastIndex,
                               cstone::DeviceVector<uint64_t>& taggedIdsIndexes);

} // namespace sphexa