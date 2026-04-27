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
 * @brief Unit tests for id tagging setting validation
 *
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"
#include "io/id_tag_setup.hpp"
#include "io/ifile_io.hpp"

TEST(IO, idTaggingGroupsInitDefaultSphereIds)
{
    unsigned int selSpheresNum = 2;
    std::vector<unsigned int> sphereGroupIds;
    unsigned int selListSize = 0;
    std::vector<unsigned int> selListGroupIds;
    sphexa::idTaggingGroupsInit(selSpheresNum, sphereGroupIds, selListSize, selListGroupIds);
    EXPECT_EQ(sphereGroupIds.size(), selSpheresNum);
    EXPECT_EQ(selListGroupIds.size(), selListSize);
    EXPECT_EQ(sphereGroupIds[0], 0);
    EXPECT_EQ(sphereGroupIds[1], 1);
}

TEST(IO, idTaggingGroupsInitDefaultSphereIdsFromListIds)
{
    unsigned int selSpheresNum = 2;
    std::vector<unsigned int> sphereGroupIds;
    unsigned int selListSize = 3;
    std::vector<unsigned int> selListGroupIds{0, 1, 2};
    sphexa::idTaggingGroupsInit(selSpheresNum, sphereGroupIds, selListSize, selListGroupIds);
    EXPECT_EQ(sphereGroupIds.size(), selSpheresNum);
    EXPECT_EQ(selListGroupIds.size(), selListSize);
    EXPECT_EQ(sphereGroupIds[0], 3);
    EXPECT_EQ(sphereGroupIds[1], 4);
}

TEST(IO, idTaggingGroupsInitDefaultListIds)
{
    unsigned int selSpheresNum = 0;
    std::vector<unsigned int> sphereGroupIds;
    unsigned int selListSize = 3;
    std::vector<unsigned int> selListGroupIds;
    sphexa::idTaggingGroupsInit(selSpheresNum, sphereGroupIds, selListSize, selListGroupIds);
    EXPECT_EQ(sphereGroupIds.size(), selSpheresNum);
    EXPECT_EQ(selListGroupIds.size(), selListSize);
    EXPECT_EQ(selListGroupIds[0], 0);
    EXPECT_EQ(selListGroupIds[1], 0);
    EXPECT_EQ(selListGroupIds[2], 0);
}

TEST(IO, idTaggingGroupsInitDefaultListIdsFromSphereIds)
{
    unsigned int selSpheresNum = 4;
    std::vector<unsigned int> sphereGroupIds{0, 1, 2, 3};
    unsigned int selListSize = 5;
    std::vector<unsigned int> selListGroupIds;
    sphexa::idTaggingGroupsInit(selSpheresNum, sphereGroupIds, selListSize, selListGroupIds);
    EXPECT_EQ(sphereGroupIds.size(), selSpheresNum);
    EXPECT_EQ(selListGroupIds.size(), selListSize);
    EXPECT_EQ(selListGroupIds[0], 4);
    EXPECT_EQ(selListGroupIds[1], 4);
    EXPECT_EQ(selListGroupIds[2], 4);
    EXPECT_EQ(selListGroupIds[3], 4);
    EXPECT_EQ(selListGroupIds[4], 4);
}


TEST(IO, idTaggingGroupsInitDefaultMixIds)
{
    unsigned int selSpheresNum = 2;
    std::vector<unsigned int> sphereGroupIds;
    unsigned int selListSize = 3;
    std::vector<unsigned int> selListGroupIds;
    sphexa::idTaggingGroupsInit(selSpheresNum, sphereGroupIds, selListSize, selListGroupIds);
    EXPECT_EQ(sphereGroupIds.size(), selSpheresNum);
    EXPECT_EQ(selListGroupIds.size(), selListSize);
    EXPECT_EQ(sphereGroupIds[0], 0);
    EXPECT_EQ(sphereGroupIds[1], 1);
    EXPECT_EQ(selListGroupIds[0], 2);
    EXPECT_EQ(selListGroupIds[1], 2);
    EXPECT_EQ(selListGroupIds[2], 2);
}

// TEST(IO, idTaggingInitSphereDefinitionSizeThrow)
// {
//     // TODO: test that an exeption is thrown when sphere data definition size is incorrect
// }

TEST(IO, idTaggingInitSphereNegativeRadiusThrow)
{
    std::vector<sphexa::IdSelectionSphere> selSpheres{sphexa::IdSelectionSphere{0.5, 0.5, 0.5, -0.1}};
    std::vector<unsigned int> sphereGroupIds;
    std::vector<uint64_t> selList;
    std::vector<unsigned int> selListGroupIds;
    EXPECT_THROW(sphexa::idTaggingSetupCheck(selSpheres, sphereGroupIds, selList, selListGroupIds, false), std::runtime_error);
}

TEST(IO, idTaggingInitSphereIdSizeThrow)
{
    std::vector<sphexa::IdSelectionSphere> selSpheres{sphexa::IdSelectionSphere{0.5, 0.5, 0.5, 0.1}, sphexa::IdSelectionSphere{0.25, 0.25, 0.25, 0.05}};
    std::vector<unsigned int> sphereGroupIds{0};
    std::vector<uint64_t> selList;
    std::vector<unsigned int> selListGroupIds;
    EXPECT_THROW(sphexa::idTaggingSetupCheck(selSpheres, sphereGroupIds, selList, selListGroupIds, false), std::runtime_error);
}

TEST(IO, idTaggingInitListDuplicateIdThrow)
{
    std::vector<sphexa::IdSelectionSphere> selSpheres;
    std::vector<unsigned int> sphereGroupIds;
    std::vector<uint64_t> selList{0, 1, 2, 2, 3};
    std::vector<unsigned int> selListGroupIds;
    EXPECT_THROW(sphexa::idTaggingSetupCheck(selSpheres, sphereGroupIds, selList, selListGroupIds, false), std::runtime_error);
}

TEST(IO, idTaggingInitListIdSizeThrow)
{
    std::vector<sphexa::IdSelectionSphere> selSpheres;
    std::vector<unsigned int> sphereGroupIds;
    std::vector<uint64_t> selList{0, 1, 2, 3};
    std::vector<unsigned int> selListGroupIds{0, 0};
    EXPECT_THROW(sphexa::idTaggingSetupCheck(selSpheres, sphereGroupIds, selList, selListGroupIds, false), std::runtime_error);
}

TEST(IO, idTaggingInitSpheresListIdDefault)
{
    std::vector<sphexa::IdSelectionSphere> selSpheres{sphexa::IdSelectionSphere{0.5, 0.5, 0.5, 0.1}};
    std::vector<unsigned int> sphereGroupIds{3};
    std::vector<uint64_t> selList{0, 1, 2, 3};
    std::vector<unsigned int> selListGroupIds;
    sphexa::idTaggingSetupCheck(selSpheres, sphereGroupIds, selList, selListGroupIds, false);
    EXPECT_EQ(selListGroupIds.size(), selList.size());
    EXPECT_EQ(selListGroupIds[0],4);
    EXPECT_EQ(selListGroupIds[1],4);
    EXPECT_EQ(selListGroupIds[2],4);
    EXPECT_EQ(selListGroupIds[3],4);
}

TEST(IO, idTaggingInitListSpheresIdDefault)
{
    std::vector<sphexa::IdSelectionSphere> selSpheres{sphexa::IdSelectionSphere{0.5, 0.5, 0.5, 0.1}, sphexa::IdSelectionSphere{0.25, 0.25, 0.25, 0.05}};
    std::vector<unsigned int> sphereGroupIds;
    std::vector<uint64_t> selList{0, 1, 2, 3};
    std::vector<unsigned int> selListGroupIds{1, 2, 3, 4};
    sphexa::idTaggingSetupCheck(selSpheres, sphereGroupIds, selList, selListGroupIds, false);
    EXPECT_EQ(sphereGroupIds.size(), selSpheres.size());
    EXPECT_EQ(sphereGroupIds[0], 5);
    EXPECT_EQ(sphereGroupIds[1], 6);
}
