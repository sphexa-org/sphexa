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
    const uint64_t first = 3;
    const uint64_t last = 10;
    std::vector<uint64_t> ids(100);
    std::iota(std::begin(ids), std::end(ids), 0);
    std::vector<uint64_t> taggedIdxRef{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::vector<uint64_t> taggedIdxRefWRange;
    std::copy_if(taggedIdxRef.begin(), taggedIdxRef.end(), std::back_inserter(taggedIdxRefWRange), [first, last](auto idx){
        return idx >= first && idx < last;
    });

    std::vector<uint64_t> taggedIdx;
    // TODO: use tagging function
    std::for_each(taggedIdxRef.begin(), taggedIdxRef.end(), [&ids = ids](auto taggedIdx){
            ids[taggedIdx] = ids[taggedIdx] | sphexa::msbMask;
    });

    sphexa::findTaggedIds(ids, 0, ids.size(), taggedIdx);
    EXPECT_EQ(taggedIdx, taggedIdxRef);

    taggedIdx.clear();
    sphexa::findTaggedIds(ids, 3, 10, taggedIdx);
    EXPECT_EQ(taggedIdx, taggedIdxRefWRange);
}