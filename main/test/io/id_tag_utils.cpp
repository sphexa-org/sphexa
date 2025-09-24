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

//TODO: some code duplication is present
//TODO: check if we are using the right types (LocalIndex, IdType)
std::vector<sphexa::IdType> makeIds(size_t n) {
    std::vector<sphexa::IdType> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    return ids;
}

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

TEST(IO, tagIdInList)
{
    std::vector<sphexa::IdType> ids = makeIds(100);
    std::vector<sphexa::IdType> selectedIds{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::vector<sphexa::IdType> tagIdsRef = ids;
    tagIdsRef[0] = 9223372036854775808ULL;
    tagIdsRef[1] = 9223372036854775809ULL;
    tagIdsRef[2] = 9223372036854775810ULL;
    tagIdsRef[3] = 9223372036854775811ULL;
    tagIdsRef[6] = 9223372036854775814ULL;
    tagIdsRef[11] = 9223372036854775819ULL;
    tagIdsRef[13] = 9223372036854775821ULL;
    tagIdsRef[23] = 9223372036854775831ULL;
    tagIdsRef[71] = 9223372036854775879ULL;
    tagIdsRef[83] = 9223372036854775891ULL;
    tagIdsRef[91] = 9223372036854775899ULL;
    tagIdsRef[95] = 9223372036854775903ULL;
    tagIdsRef[99] = 9223372036854775907ULL;

    sphexa::tagIdsInList(ids, 0, ids.size(), selectedIds);
    EXPECT_EQ(ids, tagIdsRef);
}

TEST(IO, tagIdInListWithRange)
{
    sphexa::LocalIndex first = 3;
    sphexa::LocalIndex last = 10;
    std::vector<sphexa::IdType> ids = makeIds(100);
    std::vector<sphexa::IdType> selectedIds{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::vector<sphexa::IdType> tagIdsRef = ids;
    tagIdsRef[3] = 9223372036854775811ULL;
    tagIdsRef[6] = 9223372036854775814ULL;

    sphexa::tagIdsInList(ids, first, last, selectedIds);
    EXPECT_EQ(ids, tagIdsRef);
}

TEST(IO, tagIdInSphere)
{
    std::vector<sphexa::IdType> ids = makeIds(1000);

    // Particle distribution creation
    std::vector<sphexa::CoordinateType> x, y, z;
    makeParticleDistribution(x, y, z, 1000);

    // Selection sphere definition
    sphexa::IdSelectionSphere selSphereData;
    selSphereData.radius = 0.25;
    selSphereData.center[0] = 0.0;
    selSphereData.center[1] = 0.0;
    selSphereData.center[2] = 0.0;

    std::vector<sphexa::LocalIndex> taggedIdxRef{444, 445, 454, 455, 544, 545, 554, 555};
    std::vector<sphexa::LocalIndex> taggedIdx;
    sphexa::tagIdsInSphere(ids, x, y, z, 0, ids.size(), selSphereData);
    sphexa::findTaggedIds(std::span<const sphexa::IdType>(ids), 0, ids.size(), taggedIdx);
    EXPECT_EQ(taggedIdx, taggedIdxRef);

}

TEST(IO, tagIdInSphereWithRange)
{
    std::vector<sphexa::IdType> ids = makeIds(1000);
    sphexa::LocalIndex first = 400;
    sphexa::LocalIndex last = 500;

    // Particle distribution creation
    std::vector<sphexa::CoordinateType> x, y, z;
    makeParticleDistribution(x, y, z, 1000);

    // Selection sphere definition
    sphexa::IdSelectionSphere selSphereData;
    selSphereData.radius = 0.25;
    selSphereData.center[0] = 0.0;
    selSphereData.center[1] = 0.0;
    selSphereData.center[2] = 0.0;


    std::vector<sphexa::LocalIndex> taggedIdxRef = {444, 445, 454, 455};
    std::vector<sphexa::LocalIndex> taggedIdx;
    sphexa::tagIdsInSphere(ids, x, y, z, first, last, selSphereData);
    sphexa::findTaggedIds(std::span<const sphexa::IdType>(ids), 0, ids.size(), taggedIdx);
    EXPECT_EQ(taggedIdx, taggedIdxRef);

}

TEST(IO, taggedIdIdentification)
{
    std::vector<sphexa::IdType> ids = makeIds(100);
    std::vector<sphexa::LocalIndex> taggedIdPos;

    std::vector<sphexa::LocalIndex> taggedIdPosRef{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::for_each(taggedIdPosRef.begin(), taggedIdPosRef.end(), [&ids = ids](auto idPos){
        ids[idPos] = ids[idPos] | sphexa::msbMask;
    });

    sphexa::findTaggedIds(std::span<const sphexa::IdType>(ids), 0, ids.size(), taggedIdPos);
    EXPECT_EQ(taggedIdPos, taggedIdPosRef);
}

TEST(IO, taggedIdIdentificationWithRange)
{
    std::vector<sphexa::IdType> ids = makeIds(100);
    std::vector<sphexa::LocalIndex> taggedIdPos;
    sphexa::LocalIndex first = 3;
    sphexa::LocalIndex last = 10;

    std::vector<sphexa::LocalIndex> taggedIdPosRef{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::for_each(taggedIdPosRef.begin(), taggedIdPosRef.end(), [&ids = ids](auto idPos){
        ids[idPos] = ids[idPos] | sphexa::msbMask;
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
    std::vector<sphexa::IdType> ids = makeIds(100);
    std::vector<sphexa::LocalIndex> taggedIdPos;
    sphexa::LocalIndex first = 0;
    sphexa::LocalIndex last = 3;

    std::vector<sphexa::LocalIndex> taggedIdPosRef{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::for_each(taggedIdPosRef.begin(), taggedIdPosRef.end(), [&ids = ids](auto idPos){
        ids[idPos] = ids[idPos] | sphexa::msbMask;
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
    std::vector<sphexa::IdType> ids = makeIds(100);
    std::vector<sphexa::LocalIndex> taggedIdPos;
    sphexa::LocalIndex first = 97;
    sphexa::LocalIndex last = 100;

    std::vector<sphexa::LocalIndex> taggedIdPosRef{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::for_each(taggedIdPosRef.begin(), taggedIdPosRef.end(), [&ids = ids](auto idPos){
        ids[idPos] = ids[idPos] | sphexa::msbMask;
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
    std::vector<sphexa::IdType> ids = makeIds(100);
    std::vector<sphexa::LocalIndex> taggedIdPos;

    std::vector<sphexa::LocalIndex> taggedIdPosRef = {0};
    ids[0] = ids[0] | sphexa::msbMask;

    sphexa::findTaggedIds(std::span<const sphexa::IdType>(ids), 0, ids.size(), taggedIdPos);
    EXPECT_EQ(taggedIdPos, taggedIdPosRef);
}

TEST(IO, taggedIdIdentificationSingleEnd)
{
    std::vector<sphexa::IdType> ids = makeIds(100);
    std::vector<sphexa::LocalIndex> taggedIdPos;

    std::vector<sphexa::LocalIndex> taggedIdPosRef = {99};
    ids[99] = ids[99] | sphexa::msbMask;

    sphexa::findTaggedIds(std::span<const sphexa::IdType>(ids), 0, ids.size(), taggedIdPos);
    EXPECT_EQ(taggedIdPos, taggedIdPosRef);
}