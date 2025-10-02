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
 * @brief Unit tests for id tagging related functionality
 *
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#include "io/id_tag_utils.hpp"

void makeParticleDistribution(std::vector<sphexa::CoordinateType>& x,
                              std::vector<sphexa::CoordinateType>& y,
                              std::vector<sphexa::CoordinateType>& z,
                              size_t numParticles)
{
    x.resize(numParticles);
    y.resize(numParticles);
    z.resize(numParticles);

    unsigned int gridSize = std::cbrt(numParticles);
    double step = 2.0 / (gridSize - 1);
    unsigned int index = 0;
    for (unsigned int i = 0; i < gridSize; ++i) {
        for (unsigned int j = 0; j < gridSize; ++j) {
            for (unsigned int k = 0; k < gridSize; ++k) {
                if (index >= numParticles) break;
                x[index] = -1.0 + i * step;
                y[index] = -1.0 + j * step;
                z[index] = -1.0 + k * step;
                ++index;
            }
        }
    }
}

TEST(IO, applyTaggingMaskZero)
{
    sphexa::IdType id = 0;
    sphexa::IdType groupId = 0;
    sphexa::IdType idRef = 18014398509481984ULL;
    sphexa::applyTaggingMask(groupId, id);
    EXPECT_EQ(id, idRef);
}

TEST(IO, applyTaggingMaskMaxGroup)
{
    sphexa::IdType id = 1;
    sphexa::IdType groupId = sphexa::supGroupId - 1;
    sphexa::IdType idRef = 18428729675200069633ULL;
    sphexa::applyTaggingMask(groupId, id);
    EXPECT_EQ(id, idRef);
}

TEST(IO, applyTaggingMaskMaxId)
{
    sphexa::IdType id = (sphexa::IdType(1) << (sizeof(sphexa::IdType)*8 - sphexa::taggingMaskSize)) - 1;
    sphexa::IdType groupId = 0;
    sphexa::IdType idRef = 36028797018963967;
    sphexa::applyTaggingMask(groupId, id);
    EXPECT_EQ(id, idRef);
}

TEST(IO, applyTaggingMaskMaxIdMaxGroup)
{
    sphexa::IdType id = (sphexa::IdType(1) << (sizeof(sphexa::IdType)*8 - sphexa::taggingMaskSize)) - 1;
    sphexa::IdType groupId = sphexa::supGroupId - 1;
    sphexa::IdType idRef = 18446744073709551615ULL;
    sphexa::applyTaggingMask(groupId, id);
    EXPECT_EQ(id, idRef);
}

TEST(IO, applyTaggingMaskTwice)
{
    sphexa::IdType id = 0;
    sphexa::IdType groupId = 0;
    sphexa::IdType idRef = 36028797018963968ULL;
    sphexa::applyTaggingMask(groupId, id);
    groupId = 1;
    sphexa::applyTaggingMask(groupId, id);
    EXPECT_EQ(id, idRef);
}


TEST(IO, tagIdInList)
{
    std::vector<sphexa::IdType> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<sphexa::IdType> selectedIds{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::vector<sphexa::IdType> tagIdsRef = ids;
    tagIdsRef[0] = 18014398509481984ULL;
    tagIdsRef[1] = 18014398509481985ULL;
    tagIdsRef[2] = 18014398509481986ULL;
    tagIdsRef[3] = 18014398509481987ULL;
    tagIdsRef[6] = 18014398509481990ULL;
    tagIdsRef[11] = 18014398509481995ULL;
    tagIdsRef[13] = 18014398509481997ULL;
    tagIdsRef[23] = 18014398509482007ULL;
    tagIdsRef[71] = 18014398509482055ULL;
    tagIdsRef[83] = 18014398509482067ULL;
    tagIdsRef[91] = 18014398509482075ULL;
    tagIdsRef[95] = 18014398509482079ULL;
    tagIdsRef[99] = 18014398509482083ULL;

    sphexa::tagIdsInList(std::span<sphexa::IdType>(ids), 0, ids.size(), std::span<const sphexa::IdType>(selectedIds));
    EXPECT_EQ(ids, tagIdsRef);
}

TEST(IO, tagIdInListWithRange)
{
    sphexa::LocalIndex first = 3;
    sphexa::LocalIndex last = 10;
    std::vector<sphexa::IdType> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    sphexa::IdType groupId = 1;
    std::vector<sphexa::IdType> selectedIds{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::vector<sphexa::IdType> tagIdsRef = ids;
    tagIdsRef[3] = 36028797018963971ULL;
    tagIdsRef[6] = 36028797018963974ULL;

    sphexa::tagIdsInList(std::span<sphexa::IdType>(ids), first, last, std::span<const sphexa::IdType>(selectedIds), groupId);
    EXPECT_EQ(ids, tagIdsRef);
}

TEST(IO, tagIdInSphere)
{
    std::vector<sphexa::IdType> ids(1000);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<sphexa::IdType> tagIdsRef = ids;

    // Particle distribution creation
    std::vector<sphexa::CoordinateType> x, y, z;
    makeParticleDistribution(x, y, z, 1000);

    // Selection sphere definition
    sphexa::IdSelectionSphere selSphereData{0.0, 0.0, 0.0, 0.25};

    tagIdsRef[444]=18014398509482428ULL;
    tagIdsRef[445]=18014398509482429ULL;
    tagIdsRef[454]=18014398509482438ULL;
    tagIdsRef[455]=18014398509482439ULL;
    tagIdsRef[544]=18014398509482528ULL;
    tagIdsRef[545]=18014398509482529ULL;
    tagIdsRef[554]=18014398509482538ULL;
    tagIdsRef[555]=18014398509482539ULL;
    sphexa::tagIdsInSphere(std::span<sphexa::IdType>(ids), x, y, z, 0, ids.size(), selSphereData);
    EXPECT_EQ(ids, tagIdsRef);
}

TEST(IO, tagIdInSphereWithRange)
{
    std::vector<sphexa::IdType> ids(1000);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<sphexa::IdType> tagIdsRef = ids;
    sphexa::LocalIndex first = 400;
    sphexa::LocalIndex last = 500;

    // Particle distribution creation
    std::vector<sphexa::CoordinateType> x, y, z;
    makeParticleDistribution(x, y, z, 1000);

    // Selection sphere definition
    sphexa::IdSelectionSphere selSphereData{0.0, 0.0, 0.0, 0.25};

    tagIdsRef[444]=18014398509482428ULL;
    tagIdsRef[445]=18014398509482429ULL;
    tagIdsRef[454]=18014398509482438ULL;
    tagIdsRef[455]=18014398509482439ULL;
    sphexa::tagIdsInSphere(std::span<sphexa::IdType>(ids), x, y, z, first, last, selSphereData);
    EXPECT_EQ(ids, tagIdsRef);
}

TEST(IO, taggedIdIdentification)
{
    std::vector<sphexa::IdType> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<sphexa::LocalIndex> taggedIdPos;

    std::vector<sphexa::LocalIndex> taggedIdPosRef{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::for_each(taggedIdPosRef.begin(), taggedIdPosRef.end(), [&ids = ids](auto idPos){
        sphexa::applyTaggingMask(0, ids[idPos]);
    });

    sphexa::findTaggedIds(std::span<const sphexa::IdType>(ids), 0, ids.size(), taggedIdPos);
    EXPECT_EQ(taggedIdPos, taggedIdPosRef);
}

TEST(IO, taggedIdIdentificationWithRange)
{
    std::vector<sphexa::IdType> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<sphexa::LocalIndex> taggedIdPos;
    sphexa::LocalIndex first = 3;
    sphexa::LocalIndex last = 10;

    std::vector<sphexa::LocalIndex> taggedIdPosRef{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::for_each(taggedIdPosRef.begin(), taggedIdPosRef.end(), [&ids = ids](auto idPos){
        sphexa::applyTaggingMask(1, ids[idPos]);
    });
    std::vector<sphexa::LocalIndex> taggedIdPosRefRange;
    std::copy_if(taggedIdPosRef.begin(), taggedIdPosRef.end(), std::back_inserter(taggedIdPosRefRange), [first, last](auto idPos){
        return idPos >= first && idPos < last;
    });

    sphexa::findTaggedIds(std::span<const sphexa::IdType>(ids), first, last, taggedIdPos);
    EXPECT_EQ(taggedIdPos, taggedIdPosRefRange);
}

TEST(IO, taggedIdIdentificationWithRangeStart)
{
    std::vector<sphexa::IdType> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<sphexa::LocalIndex> taggedIdPos;
    sphexa::LocalIndex first = 0;
    sphexa::LocalIndex last = 3;

    std::vector<sphexa::LocalIndex> taggedIdPosRef{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::for_each(taggedIdPosRef.begin(), taggedIdPosRef.end(), [&ids = ids](auto idPos){
        sphexa::applyTaggingMask(2, ids[idPos]);
    });
    std::vector<sphexa::LocalIndex> taggedIdPosRefRange;
    std::copy_if(taggedIdPosRef.begin(), taggedIdPosRef.end(), std::back_inserter(taggedIdPosRefRange), [first, last](auto idPos){
        return idPos >= first && idPos < last;
    });

    sphexa::findTaggedIds(std::span<const sphexa::IdType>(ids), first, last, taggedIdPos);
    EXPECT_EQ(taggedIdPos, taggedIdPosRefRange);
}

TEST(IO, taggedIdIdentificationWithRangeEnd)
{
    std::vector<sphexa::IdType> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<sphexa::LocalIndex> taggedIdPos;
    sphexa::LocalIndex first = 97;
    sphexa::LocalIndex last = 100;

    std::vector<sphexa::LocalIndex> taggedIdPosRef{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::for_each(taggedIdPosRef.begin(), taggedIdPosRef.end(), [&ids = ids](auto idPos){
        sphexa::applyTaggingMask(3, ids[idPos]);
    });
    std::vector<sphexa::LocalIndex> taggedIdPosRefRange;
    std::copy_if(taggedIdPosRef.begin(), taggedIdPosRef.end(), std::back_inserter(taggedIdPosRefRange), [first, last](auto idPos){
        return idPos >= first && idPos < last;
    });

    sphexa::findTaggedIds(std::span<const sphexa::IdType>(ids), first, last, taggedIdPos);
    EXPECT_EQ(taggedIdPos, taggedIdPosRefRange);
}

TEST(IO, taggedIdIdentificationSingleStart)
{
    std::vector<sphexa::IdType> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<sphexa::LocalIndex> taggedIdPos;

    std::vector<sphexa::LocalIndex> taggedIdPosRef = {0};
    sphexa::applyTaggingMask(4, ids[0]);

    sphexa::findTaggedIds(std::span<const sphexa::IdType>(ids), 0, ids.size(), taggedIdPos);
    EXPECT_EQ(taggedIdPos, taggedIdPosRef);
}

TEST(IO, taggedIdIdentificationSingleEnd)
{
    std::vector<sphexa::IdType> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<sphexa::LocalIndex> taggedIdPos;

    std::vector<sphexa::LocalIndex> taggedIdPosRef = {99};
    sphexa::applyTaggingMask(5, ids[99]);

    sphexa::findTaggedIds(std::span<const sphexa::IdType>(ids), 0, ids.size(), taggedIdPos);
    EXPECT_EQ(taggedIdPos, taggedIdPosRef);
}