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

#include "gtest/gtest.h"
#include "io/id_tag_utils.hpp"
 
//TODO: create tagged id list, it will be used in multiple tests
TEST(IO, taggedIdIdentificationGpu)
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

    std::for_each(taggedIdxRef.begin(), taggedIdxRef.end(), [&ids = ids](auto taggedIdx){
            ids[taggedIdx] = ids[taggedIdx] | sphexa::msbMask;
    });

    cstone::DeviceVector<uint64_t> idsDev(ids);
    sphexa::findTaggedIds(idsDev, 0, idsDev.size(), taggedIdx);
    EXPECT_EQ(taggedIdx, taggedIdxRef);

    taggedIdx.clear();
    sphexa::findTaggedIds(idsDev, 3, 10, taggedIdx);
    EXPECT_EQ(taggedIdx, taggedIdxRefWRange);
}
