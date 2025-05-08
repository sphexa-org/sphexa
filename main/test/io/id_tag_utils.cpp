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

//TODO: create tagged id list, it will be used in multiple tests


TEST(IO, tagIdInList)
{
    const uint64_t first = 3;
    const uint64_t last = 10;
    std::vector<uint64_t> ids(100);
    std::iota(std::begin(ids), std::end(ids), 0);
    std::vector<uint64_t> selectedIds{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::vector<uint64_t> tagIdsRef = ids;
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

    std::iota(std::begin(ids), std::end(ids), 0);
    tagIdsRef = ids;
    tagIdsRef[3] = 9223372036854775811ULL;
    tagIdsRef[6] = 9223372036854775814ULL;

    sphexa::tagIdsInList(ids, first, last, selectedIds);
    EXPECT_EQ(ids, tagIdsRef);
}

TEST(IO, tagIdInSphere)
{
    const uint64_t first = 400;
    const uint64_t last = 500;
    std::vector<uint64_t> ids(1000);
    std::iota(std::begin(ids), std::end(ids), 0);

    // Particle distribution creation
    std::vector<sphexa::CoordinateType> x(1000);
    std::vector<sphexa::CoordinateType> y(1000);
    std::vector<sphexa::CoordinateType> z(1000);
    unsigned int gridSize = std::cbrt(1000);
    double step = 2.0 / (gridSize - 1);
    int index = 0;
    for (int i=0; i<10; ++i) {
        for (int j=0; j<10; ++j) {
            for (int k=0; k<10; ++k) {
                x[index] = -1.0 + i*step;
                y[index] = -1.0 + j*step;
                z[index] = -1.0 + k*step;
                ++index;
            }
        }
    }

    // Selection sphere definition
    sphexa::IdSelectionSphere selSphereData;
    selSphereData.radius = 0.25;
    selSphereData.center[0] = 0.0;
    selSphereData.center[1] = 0.0;
    selSphereData.center[2] = 0.0;

    std::vector<uint64_t> taggedIdxRef{444, 445, 454, 455, 544, 545, 554, 555};
    std::vector<uint64_t> taggedIdx;
    sphexa::tagIdsInSphere(ids, x, y, z, 0, ids.size(), selSphereData);
    sphexa::findTaggedIds(ids, 0, ids.size(), taggedIdx);
    EXPECT_EQ(taggedIdx, taggedIdxRef);

    taggedIdxRef = {444, 445, 454, 455};
    std::iota(std::begin(ids), std::end(ids), 0);
    taggedIdx.clear();
    sphexa::tagIdsInSphere(ids, x, y, z, first, last, selSphereData);
    sphexa::findTaggedIds(ids, 0, ids.size(), taggedIdx);
    EXPECT_EQ(taggedIdx, taggedIdxRef);

}

TEST(IO, taggedIdIdentification)
{
    uint64_t first;
    uint64_t last;
    std::vector<uint64_t> ids(100);
    std::vector<uint64_t> taggedIdPos;

    // Full range test 1
    std::vector<uint64_t> taggedIdPosRef{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::iota(std::begin(ids), std::end(ids), 0);
    std::for_each(taggedIdPosRef.begin(), taggedIdPosRef.end(), [&ids = ids](auto idPos){
        ids[idPos] = ids[idPos] | sphexa::msbMask;
    });
    sphexa::findTaggedIds(ids, 0, ids.size(), taggedIdPos);
    EXPECT_EQ(taggedIdPos, taggedIdPosRef);

    // Range test 1
    first = 3;
    last = 10;
    std::vector<uint64_t> taggedIdPosRefRange;
    std::copy_if(taggedIdPosRef.begin(), taggedIdPosRef.end(), std::back_inserter(taggedIdPosRefRange), [first, last](auto idPos){
        return idPos >= first && idPos < last;
    });
    sphexa::findTaggedIds(ids, first, last, taggedIdPos);
    EXPECT_EQ(taggedIdPos, taggedIdPosRefRange);

    // Range test 2
    first = 0;
    last = 100;
    taggedIdPosRefRange.clear();
    std::copy_if(taggedIdPosRef.begin(), taggedIdPosRef.end(), std::back_inserter(taggedIdPosRefRange), [first, last](auto idPos){
        return idPos >= first && idPos < last;
    });
    sphexa::findTaggedIds(ids, first, last, taggedIdPos);
    EXPECT_EQ(taggedIdPos, taggedIdPosRefRange);

    // Range test 3
    first = 0;
    last = 1;
    taggedIdPosRefRange.clear();
    std::copy_if(taggedIdPosRef.begin(), taggedIdPosRef.end(), std::back_inserter(taggedIdPosRefRange), [first, last](auto idPos){
        return idPos >= first && idPos < last;
    });
    sphexa::findTaggedIds(ids, first, last, taggedIdPos);
    EXPECT_EQ(taggedIdPos, taggedIdPosRefRange);

    // Range test 4
    first = 99;
    last = 100;
    taggedIdPosRefRange.clear();
    std::copy_if(taggedIdPosRef.begin(), taggedIdPosRef.end(), std::back_inserter(taggedIdPosRefRange), [first, last](auto idPos){
        return idPos >= first && idPos < last;
    });
    sphexa::findTaggedIds(ids, first, last, taggedIdPos);
    EXPECT_EQ(taggedIdPos, taggedIdPosRefRange);

    // Full range test 2
    taggedIdPosRef = {0};
    std::iota(std::begin(ids), std::end(ids), 0);
    std::for_each(taggedIdPosRef.begin(), taggedIdPosRef.end(), [&ids = ids](auto idPos){
        ids[idPos] = ids[idPos] | sphexa::msbMask;
    });
    sphexa::findTaggedIds(ids, 0, ids.size(), taggedIdPos);
    EXPECT_EQ(taggedIdPos, taggedIdPosRef);

    // Full range test 3
    taggedIdPosRef = {99};
    std::iota(std::begin(ids), std::end(ids), 0);
    std::for_each(taggedIdPosRef.begin(), taggedIdPosRef.end(), [&ids = ids](auto idPos){
        ids[idPos] = ids[idPos] | sphexa::msbMask;
    });
    sphexa::findTaggedIds(ids, 0, ids.size(), taggedIdPos);
    EXPECT_EQ(taggedIdPos, taggedIdPosRef);

}