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

#include <vector>

#include <thrust/device_vector.h>

#include "gtest/gtest.h"
#include "io/id_tag_utils.hpp"

TEST(IO, taggedIdIdentificationGPU)
{
    std::vector<sphexa::IdType> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<sphexa::LocalIndex> taggedIdPos;

    std::vector<sphexa::LocalIndex> taggedIdPosRef{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::for_each(taggedIdPosRef.begin(), taggedIdPosRef.end(), [&ids = ids](auto idPos){
        sphexa::applyTaggingMask(0, ids[idPos]);
    });
    thrust::device_vector<sphexa::IdType> idsDev(ids);

    sphexa::findTaggedIdsGPU(std::span<const sphexa::IdType>(thrust::raw_pointer_cast(idsDev.data()), idsDev.size()), 0, idsDev.size(), taggedIdPos);
    EXPECT_EQ(taggedIdPos, taggedIdPosRef);
}

TEST(IO, taggedIdIdentificationWithRangeGPU)
{
    std::vector<sphexa::IdType> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<sphexa::LocalIndex> taggedIdPos;
    sphexa::LocalIndex first = 3;
    sphexa::LocalIndex last = 10;

    std::vector<uint32_t> taggedIdPosRef{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::vector<uint32_t> taggedIdPosRefRange;
    std::copy_if(taggedIdPosRef.begin(), taggedIdPosRef.end(), std::back_inserter(taggedIdPosRefRange), [first, last](auto idPos){
        return idPos >= first && idPos < last;
    });
    std::for_each(taggedIdPosRef.begin(), taggedIdPosRef.end(), [&ids = ids](auto idPos){
        sphexa::applyTaggingMask(1, ids[idPos]);
    });
    thrust::device_vector<sphexa::IdType> idsDev(ids);

    sphexa::findTaggedIdsGPU(std::span<const sphexa::IdType>(thrust::raw_pointer_cast(idsDev.data()), idsDev.size()), first, last, taggedIdPos);
    EXPECT_EQ(taggedIdPos, taggedIdPosRefRange);
}

TEST(IO, taggedIdIdentificationWithRangeStartGPU)
{
    std::vector<sphexa::IdType> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<sphexa::LocalIndex> taggedIdPos;
    sphexa::LocalIndex first = 0;
    sphexa::LocalIndex last = 3;

    std::vector<uint32_t> taggedIdPosRef{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::vector<uint32_t> taggedIdPosRefRange;
    std::copy_if(taggedIdPosRef.begin(), taggedIdPosRef.end(), std::back_inserter(taggedIdPosRefRange), [first, last](auto idPos){
        return idPos >= first && idPos < last;
    });
    std::for_each(taggedIdPosRef.begin(), taggedIdPosRef.end(), [&ids = ids](auto idPos){
        sphexa::applyTaggingMask(2, ids[idPos]);
    });
    thrust::device_vector<sphexa::IdType> idsDev(ids);

    sphexa::findTaggedIdsGPU(std::span<const sphexa::IdType>(thrust::raw_pointer_cast(idsDev.data()), idsDev.size()), first, last, taggedIdPos);
    EXPECT_EQ(taggedIdPos, taggedIdPosRefRange);
}

TEST(IO, taggedIdIdentificationWithRangeEndGPU)
{
    std::vector<sphexa::IdType> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<sphexa::LocalIndex> taggedIdPos;
    sphexa::LocalIndex first = 97;
    sphexa::LocalIndex last = 100;

    std::vector<uint32_t> taggedIdPosRef{0, 1, 2, 3, 6, 11, 13, 23, 71, 83, 91, 95, 99};
    std::vector<uint32_t> taggedIdPosRefRange;
    std::copy_if(taggedIdPosRef.begin(), taggedIdPosRef.end(), std::back_inserter(taggedIdPosRefRange), [first, last](auto idPos){
        return idPos >= first && idPos < last;
    });
    std::for_each(taggedIdPosRef.begin(), taggedIdPosRef.end(), [&ids = ids](auto idPos){
        sphexa::applyTaggingMask(3, ids[idPos]);
    });
    thrust::device_vector<sphexa::IdType> idsDev(ids);

    sphexa::findTaggedIdsGPU(std::span<const sphexa::IdType>(thrust::raw_pointer_cast(idsDev.data()), idsDev.size()), first, last, taggedIdPos);
    EXPECT_EQ(taggedIdPos, taggedIdPosRefRange);
}

TEST(IO, taggedIdIdentificationSingleStartGPU)
{
    std::vector<sphexa::IdType> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<sphexa::LocalIndex> taggedIdPos;

    std::vector<uint32_t> taggedIdPosRef{0};
    sphexa::applyTaggingMask(4, ids[0]);
    thrust::device_vector<sphexa::IdType> idsDev(ids);

    sphexa::findTaggedIdsGPU(std::span<const sphexa::IdType>(thrust::raw_pointer_cast(idsDev.data()), idsDev.size()), 0, idsDev.size(), taggedIdPos);
    EXPECT_EQ(taggedIdPos, taggedIdPosRef);
}

TEST(IO, taggedIdIdentificationSingleEndGPU)
{
    std::vector<sphexa::IdType> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<sphexa::LocalIndex> taggedIdPos;

    std::vector<uint32_t> taggedIdPosRef{99};
    sphexa::applyTaggingMask(5, ids[99]);
    thrust::device_vector<sphexa::IdType> idsDev(ids);

    sphexa::findTaggedIdsGPU(std::span<const sphexa::IdType>(thrust::raw_pointer_cast(idsDev.data()), idsDev.size()), 0, idsDev.size(), taggedIdPos);
    EXPECT_EQ(taggedIdPos, taggedIdPosRef);
}
