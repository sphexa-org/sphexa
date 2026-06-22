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
 * @brief Unit tests for id tagging related functionality, GPU version
 *
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 */

#include <algorithm>
#include <numeric>
#include <vector>

#include <thrust/device_vector.h>

#include "id_tag_utils.hpp"
#include "cstone/cuda/device_vector.h"
#include "cstone/cuda/cuda_utils.cuh"
#include "gtest/gtest.h"
#include "io/id_tag_utils.hpp"

TEST(IO, tagIdInListGPU)
{
    std::vector<uint64_t> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<uint64_t>   selectedIds{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::vector<unsigned>   selectedIdsGroups(selectedIds.size(), 0);
    std::vector<uint64_t>   tagIdsRef = ids;
    tagIdsRef[0]                      = 18014398509481984ULL;
    tagIdsRef[1]                      = 18014398509481985ULL;
    tagIdsRef[2]                      = 18014398509481986ULL;
    tagIdsRef[3]                      = 18014398509481987ULL;
    tagIdsRef[6]                      = 18014398509481990ULL;
    tagIdsRef[11]                     = 18014398509481995ULL;
    tagIdsRef[13]                     = 18014398509481997ULL;
    tagIdsRef[23]                     = 18014398509482007ULL;
    tagIdsRef[71]                     = 18014398509482055ULL;
    tagIdsRef[83]                     = 18014398509482067ULL;
    tagIdsRef[91]                     = 18014398509482075ULL;
    tagIdsRef[95]                     = 18014398509482079ULL;
    tagIdsRef[99]                     = 18014398509482083ULL;

    cstone::DeviceVector<uint64_t> idsDev(ids);
    sphexa::tagIdsInListGPU(std::span<uint64_t>(thrust::raw_pointer_cast(idsDev.data()), idsDev.size()), 0, idsDev.size(),
                            std::span<const uint64_t>(selectedIds),
                            std::span<const unsigned>(selectedIdsGroups));

    EXPECT_EQ(toHost(idsDev), tagIdsRef);
}

TEST(IO, tagIdInListMultipleGroupsGPU)
{
    std::vector<uint64_t> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<uint64_t>   selectedIds{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::vector<unsigned>   selectedIdsGroups{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2};
    std::vector<uint64_t>   tagIdsRef = ids;
    tagIdsRef[0]                      = 18014398509481984ULL;
    tagIdsRef[1]                      = 18014398509481985ULL;
    tagIdsRef[2]                      = 18014398509481986ULL;
    tagIdsRef[3]                      = 18014398509481987ULL;
    tagIdsRef[6]                      = 36028797018963974ULL;
    tagIdsRef[11]                     = 36028797018963979ULL;
    tagIdsRef[13]                     = 36028797018963981ULL;
    tagIdsRef[23]                     = 36028797018963991ULL;
    tagIdsRef[71]                     = 54043195528446023ULL;
    tagIdsRef[83]                     = 54043195528446035ULL;
    tagIdsRef[91]                     = 54043195528446043ULL;
    tagIdsRef[95]                     = 54043195528446047ULL;
    tagIdsRef[99]                     = 54043195528446051ULL;

    cstone::DeviceVector<uint64_t> idsDev(ids);
    sphexa::tagIdsInListGPU(std::span<uint64_t>(thrust::raw_pointer_cast(idsDev.data()), idsDev.size()), 0, idsDev.size(),
                            std::span<const uint64_t>(selectedIds),
                            std::span<const unsigned>(selectedIdsGroups));

    EXPECT_EQ(toHost(idsDev), tagIdsRef);
}

TEST(IO, tagIdInListWithRangeGPU)
{
    uint32_t              first = 3;
    uint32_t              last  = 10;
    std::vector<uint64_t> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<uint64_t> selectedIds{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::vector<unsigned> selectedIdsGroups(selectedIds.size(), 0);
    std::vector<uint64_t>   tagIdsRef = ids;
    tagIdsRef[3]                      = 18014398509481987ULL;
    tagIdsRef[6]                      = 18014398509481990ULL;

    cstone::DeviceVector<uint64_t> idsDev(ids);
    sphexa::tagIdsInListGPU(std::span<uint64_t>(thrust::raw_pointer_cast(idsDev.data()), idsDev.size()), first, last,
                            std::span<const uint64_t>(selectedIds),
                            std::span<const unsigned>(selectedIdsGroups));
    EXPECT_EQ(toHost(idsDev), tagIdsRef);
}

TEST(IO, tagIdInListWithRangeMultipleGroupsGPU)
{
    uint32_t              first = 3;
    uint32_t              last  = 94;
    std::vector<uint64_t> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<uint64_t> selectedIds{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::vector<unsigned> selectedIdsGroups{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2};
    std::vector<uint64_t>   tagIdsRef = ids;
    tagIdsRef[3]                      = 18014398509481987ULL;
    tagIdsRef[6]                      = 36028797018963974ULL;
    tagIdsRef[11]                     = 36028797018963979ULL;
    tagIdsRef[13]                     = 36028797018963981ULL;
    tagIdsRef[23]                     = 36028797018963991ULL;
    tagIdsRef[71]                     = 54043195528446023ULL;
    tagIdsRef[83]                     = 54043195528446035ULL;
    tagIdsRef[91]                     = 54043195528446043ULL;

    cstone::DeviceVector<uint64_t> idsDev(ids);
    sphexa::tagIdsInListGPU(std::span<uint64_t>(thrust::raw_pointer_cast(idsDev.data()), idsDev.size()), first, last,
                            std::span<const uint64_t>(selectedIds),
                            std::span<const unsigned>(selectedIdsGroups));

    EXPECT_EQ(toHost(idsDev), tagIdsRef);
}

TEST(IO, tagIdInSphereGPU)
{
    std::vector<uint64_t> ids(1000);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<uint64_t> tagIdsRef = ids;

    // Particle distribution creation
    std::vector<sphexa::CoordinateType> x, y, z;
    makeParticleDistribution(x, y, z, 1000);

    // Selection sphere definition
    sphexa::IdSelectionSphere              selSphereData{0.0, 0.0, 0.0, 0.25};
    std::vector<sphexa::IdSelectionSphere> selSphereDataVec{selSphereData};

    tagIdsRef[444] = 18014398509482428ULL;
    tagIdsRef[445] = 18014398509482429ULL;
    tagIdsRef[454] = 18014398509482438ULL;
    tagIdsRef[455] = 18014398509482439ULL;
    tagIdsRef[544] = 18014398509482528ULL;
    tagIdsRef[545] = 18014398509482529ULL;
    tagIdsRef[554] = 18014398509482538ULL;
    tagIdsRef[555] = 18014398509482539ULL;

    cstone::DeviceVector<uint64_t> idsDev(ids);
    cstone::DeviceVector<sphexa::CoordinateType> xDev(x);
    cstone::DeviceVector<sphexa::CoordinateType> yDev(y);
    cstone::DeviceVector<sphexa::CoordinateType> zDev(z);
    sphexa::tagIdsInSphereGPU(std::span<uint64_t>(thrust::raw_pointer_cast(idsDev.data()), idsDev.size()),
                              std::span<const sphexa::CoordinateType>(thrust::raw_pointer_cast(xDev.data()), xDev.size()),
                              std::span<const sphexa::CoordinateType>(thrust::raw_pointer_cast(yDev.data()), yDev.size()),
                              std::span<const sphexa::CoordinateType>(thrust::raw_pointer_cast(zDev.data()), zDev.size()),
                              0, idsDev.size(), std::span<const sphexa::IdSelectionSphere>(selSphereDataVec),
                              std::span<const unsigned>(std::vector<unsigned>{0}));
    EXPECT_EQ(toHost(idsDev), tagIdsRef);
}

TEST(IO, tagIdInSphereWithRangeGPU)
{
    std::vector<uint64_t> ids(1000);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<uint64_t> tagIdsRef = ids;
    uint32_t              first     = 400;
    uint32_t              last      = 500;

    // Particle distribution creation
    std::vector<sphexa::CoordinateType> x, y, z;
    makeParticleDistribution(x, y, z, 1000);

    // Selection sphere definition
    sphexa::IdSelectionSphere              selSphereData{0.0, 0.0, 0.0, 0.25};
    std::vector<sphexa::IdSelectionSphere> selSphereDataVec{selSphereData};

    tagIdsRef[444] = 18014398509482428ULL;
    tagIdsRef[445] = 18014398509482429ULL;
    tagIdsRef[454] = 18014398509482438ULL;
    tagIdsRef[455] = 18014398509482439ULL;
    cstone::DeviceVector<uint64_t> idsDev(ids);
    cstone::DeviceVector<sphexa::CoordinateType> xDev(x);
    cstone::DeviceVector<sphexa::CoordinateType> yDev(y);
    cstone::DeviceVector<sphexa::CoordinateType> zDev(z);
    sphexa::tagIdsInSphereGPU(std::span<uint64_t>(thrust::raw_pointer_cast(idsDev.data()), idsDev.size()),
                              std::span<const sphexa::CoordinateType>(thrust::raw_pointer_cast(xDev.data()), xDev.size()),
                              std::span<const sphexa::CoordinateType>(thrust::raw_pointer_cast(yDev.data()), yDev.size()),
                              std::span<const sphexa::CoordinateType>(thrust::raw_pointer_cast(zDev.data()), zDev.size()),
                              first, last, std::span<const sphexa::IdSelectionSphere>(selSphereDataVec),
                              std::span<const unsigned>(std::vector<unsigned>{0}));
    EXPECT_EQ(toHost(idsDev), tagIdsRef);
}

TEST(IO, tagIdInMultipleSpheresGPU)
{
    std::vector<uint64_t> ids(1000);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<uint64_t> tagIdsRef = ids;

    // Particle distribution creation
    std::vector<sphexa::CoordinateType> x, y, z;
    makeParticleDistribution(x, y, z, 1000);

    // Selection spheres definition: second sphere overlaps with first one
    std::vector<sphexa::IdSelectionSphere> selSphereDataVec{sphexa::IdSelectionSphere{0.0, 0.0, 0.0, 0.25},
                                                            sphexa::IdSelectionSphere{0.1, 0.1, 0.1, 0.25}};

    tagIdsRef[444] = 18014398509482428ULL;
    tagIdsRef[445] = 18014398509482429ULL;
    tagIdsRef[454] = 18014398509482438ULL;
    tagIdsRef[455] = 36028797018964423ULL;
    tagIdsRef[544] = 18014398509482528ULL;
    tagIdsRef[545] = 36028797018964513ULL;
    tagIdsRef[554] = 36028797018964522ULL;
    tagIdsRef[555] = 36028797018964523ULL;
    tagIdsRef[556] = 36028797018964524ULL;
    tagIdsRef[565] = 36028797018964533ULL;
    tagIdsRef[655] = 36028797018964623ULL;

    cstone::DeviceVector<uint64_t> idsDev(ids);
    cstone::DeviceVector<sphexa::CoordinateType> xDev(x);
    cstone::DeviceVector<sphexa::CoordinateType> yDev(y);
    cstone::DeviceVector<sphexa::CoordinateType> zDev(z);
    sphexa::tagIdsInSphereGPU(std::span<uint64_t>(thrust::raw_pointer_cast(idsDev.data()), idsDev.size()),
                              std::span<const sphexa::CoordinateType>(thrust::raw_pointer_cast(xDev.data()), xDev.size()),
                              std::span<const sphexa::CoordinateType>(thrust::raw_pointer_cast(yDev.data()), yDev.size()),
                              std::span<const sphexa::CoordinateType>(thrust::raw_pointer_cast(zDev.data()), zDev.size()),
                              0, idsDev.size(), std::span<const sphexa::IdSelectionSphere>(selSphereDataVec),
                              std::span<const unsigned>(std::vector<unsigned>{0, 1}));
    EXPECT_EQ(toHost(idsDev), tagIdsRef);
}

TEST(IO, taggedIdIdentificationGPU)
{
    std::vector<uint64_t> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    cstone::DeviceVector<uint32_t> taggedIdPosDev;

    std::vector<uint32_t> taggedIdPosRef{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::for_each(taggedIdPosRef.begin(), taggedIdPosRef.end(),
                  [&ids = ids](auto idPos) { ids[idPos] = sphexa::applyTaggingMask(0, ids[idPos]); });
    thrust::device_vector<uint64_t> idsDev(ids);
    sphexa::findTaggedIdsGPU(std::span<const uint64_t>(thrust::raw_pointer_cast(idsDev.data()), idsDev.size()), 0,
                             idsDev.size(), taggedIdPosDev);
    EXPECT_EQ(toHost(taggedIdPosDev), taggedIdPosRef);
}

TEST(IO, taggedIdIdentificationWithRangeGPU)
{
    std::vector<uint64_t> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    cstone::DeviceVector<uint32_t> taggedIdPosDev;
    uint32_t              first = 3;
    uint32_t              last  = 10;

    std::vector<uint32_t> taggedIdPosRef{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::vector<uint32_t> taggedIdPosRefRange;
    std::copy_if(taggedIdPosRef.begin(), taggedIdPosRef.end(), std::back_inserter(taggedIdPosRefRange),
                 [first, last](auto idPos) { return idPos >= first && idPos < last; });
    std::for_each(taggedIdPosRef.begin(), taggedIdPosRef.end(),
                  [&ids = ids](auto idPos) { ids[idPos] = sphexa::applyTaggingMask(1, ids[idPos]); });
    thrust::device_vector<uint64_t> idsDev(ids);

    sphexa::findTaggedIdsGPU(std::span<const uint64_t>(thrust::raw_pointer_cast(idsDev.data()), idsDev.size()), first,
                             last, taggedIdPosDev);
    EXPECT_EQ(toHost(taggedIdPosDev), taggedIdPosRefRange);
}

TEST(IO, taggedIdIdentificationWithRangeStartGPU)
{
    std::vector<uint64_t> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    cstone::DeviceVector<uint32_t> taggedIdPosDev;
    uint32_t              first = 0;
    uint32_t              last  = 3;

    std::vector<uint32_t> taggedIdPosRef{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::vector<uint32_t> taggedIdPosRefRange;
    std::copy_if(taggedIdPosRef.begin(), taggedIdPosRef.end(), std::back_inserter(taggedIdPosRefRange),
                 [first, last](auto idPos) { return idPos >= first && idPos < last; });
    std::for_each(taggedIdPosRef.begin(), taggedIdPosRef.end(),
                  [&ids = ids](auto idPos) { ids[idPos] = sphexa::applyTaggingMask(2, ids[idPos]); });
    thrust::device_vector<uint64_t> idsDev(ids);

    sphexa::findTaggedIdsGPU(std::span<const uint64_t>(thrust::raw_pointer_cast(idsDev.data()), idsDev.size()), first,
                             last, taggedIdPosDev);
    EXPECT_EQ(toHost(taggedIdPosDev), taggedIdPosRefRange);
}

TEST(IO, taggedIdIdentificationWithRangeEndGPU)
{
    std::vector<uint64_t> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    cstone::DeviceVector<uint32_t> taggedIdPosDev;
    uint32_t              first = 97;
    uint32_t              last  = 100;

    std::vector<uint32_t> taggedIdPosRef{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::vector<uint32_t> taggedIdPosRefRange;
    std::copy_if(taggedIdPosRef.begin(), taggedIdPosRef.end(), std::back_inserter(taggedIdPosRefRange),
                 [first, last](auto idPos) { return idPos >= first && idPos < last; });
    std::for_each(taggedIdPosRef.begin(), taggedIdPosRef.end(),
                  [&ids = ids](auto idPos) { ids[idPos] = sphexa::applyTaggingMask(3, ids[idPos]); });
    thrust::device_vector<uint64_t> idsDev(ids);

    sphexa::findTaggedIdsGPU(std::span<const uint64_t>(thrust::raw_pointer_cast(idsDev.data()), idsDev.size()), first,
                             last, taggedIdPosDev);
    EXPECT_EQ(toHost(taggedIdPosDev), taggedIdPosRefRange);
}

TEST(IO, taggedIdIdentificationSingleStartGPU)
{
    std::vector<uint64_t> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    cstone::DeviceVector<uint32_t> taggedIdPosDev;

    std::vector<uint32_t> taggedIdPosRef{0};
    ids[0] = sphexa::applyTaggingMask(4, ids[0]);
    thrust::device_vector<uint64_t> idsDev(ids);

    sphexa::findTaggedIdsGPU(std::span<const uint64_t>(thrust::raw_pointer_cast(idsDev.data()), idsDev.size()), 0,
                             idsDev.size(), taggedIdPosDev);
    EXPECT_EQ(toHost(taggedIdPosDev), taggedIdPosRef);
}

TEST(IO, taggedIdIdentificationSingleEndGPU)
{
    std::vector<uint64_t> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    cstone::DeviceVector<uint32_t> taggedIdPosDev;

    std::vector<uint32_t> taggedIdPosRef{99};
    ids[99] = sphexa::applyTaggingMask(5, ids[99]);
    thrust::device_vector<uint64_t> idsDev(ids);

    sphexa::findTaggedIdsGPU(std::span<const uint64_t>(thrust::raw_pointer_cast(idsDev.data()), idsDev.size()), 0,
                             idsDev.size(), taggedIdPosDev);
    EXPECT_EQ(toHost(taggedIdPosDev), taggedIdPosRef);
}
