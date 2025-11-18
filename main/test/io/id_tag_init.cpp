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
#include "init/settings.hpp"

TEST(IO, idTaggingInitDefaultSphereIds)
{
    sphexa::InitSettings settings;
    settings["w_subset"] = 10;
    settings["id_selection_spheres"] = sphexa::VectorValue{0.5, 0.5, 0.5, 0.1, 0.25, 0.25, 0.25, 0.05};
    sphexa::idTaggingSetupInit(settings);

    EXPECT_EQ(settings.at("id_selection_spheres_group_ids").size(), 2);
    EXPECT_EQ(settings.at("id_selection_spheres_group_ids").data()[0],0);
    EXPECT_EQ(settings.at("id_selection_spheres_group_ids").data()[1],1);
}

TEST(IO, idTaggingInitSphereDefinitionSizeThrow)
{
    sphexa::InitSettings settings;
    settings["w_subset"] = 10;
    settings["id_selection_spheres"] = sphexa::VectorValue{0.5, 0.5, 0.5, 0.1, 0.25};
    EXPECT_THROW(sphexa::idTaggingSetupInit(settings), std::runtime_error);
}

TEST(IO, idTaggingInitSphereNegativeRadiusThrow)
{
    sphexa::InitSettings settings;
    settings["w_subset"] = 10;
    settings["id_selection_spheres"] = sphexa::VectorValue{0.5, 0.5, 0.5, -0.1};
    EXPECT_THROW(sphexa::idTaggingSetupInit(settings), std::runtime_error);
}

TEST(IO, idTaggingInitSphereIdSizeThrow)
{
    sphexa::InitSettings settings;
    settings["w_subset"] = 10;
    settings["id_selection_spheres"] = sphexa::VectorValue{0.5, 0.5, 0.5, 0.1, 0.25, 0.25, 0.25, 0.05};
    settings["id_selection_spheres_group_ids"] = sphexa::VectorValue{0, 1, 2};
    EXPECT_THROW(sphexa::idTaggingSetupInit(settings), std::runtime_error);
}

TEST(IO, idTaggingInitSphereNegativeIdThrow)
{
    sphexa::InitSettings settings;
    settings["w_subset"] = 10;
    settings["id_selection_spheres"] = sphexa::VectorValue{0.5, 0.5, 0.5, 0.1, 0.25, 0.25, 0.25, 0.05};
    settings["id_selection_spheres_group_ids"] = sphexa::VectorValue{-1, -2};
    EXPECT_THROW(sphexa::idTaggingSetupInit(settings), std::runtime_error);
}

TEST(IO, idTaggingInitDefaultListIds)
{
    sphexa::InitSettings settings;
    settings["w_subset"] = 10;
    settings["id_selection_list"] = sphexa::VectorValue{0, 1, 2, 3};
    sphexa::idTaggingSetupInit(settings);

    EXPECT_EQ(settings.at("id_selection_spheres_group_ids").size(), settings["id_selection_list"].size());
    EXPECT_EQ(settings.at("id_selection_spheres_group_ids").data()[0],0);
    EXPECT_EQ(settings.at("id_selection_spheres_group_ids").data()[1],0);
    EXPECT_EQ(settings.at("id_selection_spheres_group_ids").data()[2],0);
    EXPECT_EQ(settings.at("id_selection_spheres_group_ids").data()[3],0);
}

TEST(IO, idTaggingInitListDuplicateIdThrow)
{
    sphexa::InitSettings settings;
    settings["w_subset"] = 10;
    settings["id_selection_list"] = sphexa::VectorValue{0, 1, 2, 3, 3};
    EXPECT_THROW(sphexa::idTaggingSetupInit(settings), std::runtime_error);
}

TEST(IO, idTaggingInitListIdSizeThrow)
{
    sphexa::InitSettings settings;
    settings["w_subset"] = 10;
    settings["id_selection_list"] = sphexa::VectorValue{0, 1, 2, 3};
    settings["id_selection_list_group_ids"] = sphexa::VectorValue{0, 0};
    EXPECT_THROW(sphexa::idTaggingSetupInit(settings), std::runtime_error);
}

TEST(IO, idTaggingInitSpheresListIdDefault)
{
    sphexa::InitSettings settings;
    settings["w_subset"] = 10;
    settings["id_selection_spheres"] = sphexa::VectorValue{0.5, 0.5, 0.5, 0.1};
    settings["id_selection_spheres_group_ids"] = sphexa::VectorValue{3};
    settings["id_selection_list"] = sphexa::VectorValue{0, 1, 2, 3};
    sphexa::idTaggingSetupInit(settings);
    EXPECT_EQ(settings.at("id_selection_list_group_ids").size(), settings["id_selection_list"].size());
    EXPECT_EQ(settings.at("id_selection_list_group_ids").data()[0],4);
    EXPECT_EQ(settings.at("id_selection_list_group_ids").data()[1],4);
    EXPECT_EQ(settings.at("id_selection_list_group_ids").data()[2],4);
    EXPECT_EQ(settings.at("id_selection_list_group_ids").data()[3],4);

    settings.erase("id_selection_spheres_group_ids");
    settings["id_selection_list_group_ids"] = sphexa::VectorValue{1, 2, 3, 4};
}

TEST(IO, idTaggingInitListSpheresIdDefault)
{
    sphexa::InitSettings settings;
    settings["w_subset"] = 10;
    settings["id_selection_spheres"] = sphexa::VectorValue{0.5, 0.5, 0.5, 0.1, 0.25, 0.25, 0.25, 0.05};
    settings["id_selection_list"] = sphexa::VectorValue{0, 1, 2, 3};
    settings["id_selection_list_group_ids"] = sphexa::VectorValue{1, 2, 3, 4};
    sphexa::idTaggingSetupInit(settings);
    EXPECT_EQ(settings.at("id_selection_spheres_group_ids").size(), settings["id_selection_spheres"].size()/4);
    EXPECT_EQ(settings.at("id_selection_list_group_ids").data()[0],5);
    EXPECT_EQ(settings.at("id_selection_list_group_ids").data()[1],6);
}

TEST(IO, idTaggingInactive)
{
    sphexa::InitSettings settings;
    const std::string initCond = "dummy_cond";
    const std::string outputFileSuffix = ".dummy_suffix";
    std::string outFileSubset;
    std::string writeFreqStrSubset;
    std::vector<std::string> outputFieldsSubset;
    std::vector<std::string> writeExtraSubset;

    EXPECT_EQ(sphexa::idTaggingOutputParameterRetrieval(settings, initCond, outputFileSuffix, 
        outFileSubset, writeFreqStrSubset, outputFieldsSubset, writeExtraSubset), false);
}

TEST(IO, idTaggingWriteFrequencyRetrievalInteger)
{
    sphexa::InitSettings settings;
    settings["w_subset"] = 10;
    const std::string initCond = "dummy_cond";
    const std::string outputFileSuffix = ".dummy_suffix";
    std::string outFileSubset;
    std::string writeFreqStrSubset;
    std::vector<std::string> outputFieldsSubset;
    std::vector<std::string> writeExtraSubset;

    EXPECT_EQ(sphexa::idTaggingOutputParameterRetrieval(settings, initCond, outputFileSuffix, 
        outFileSubset, writeFreqStrSubset, outputFieldsSubset, writeExtraSubset), true);

    EXPECT_EQ(writeFreqStrSubset, "10");
}

TEST(IO, idTaggingWriteFrequencyRetrievalFloatingPoint)
{
    sphexa::InitSettings settings;
    settings["w_subset"] = 14.83609;
    const std::string initCond = "dummy_cond";
    const std::string outputFileSuffix = ".dummy_suffix";
    std::string outFileSubset;
    std::string writeFreqStrSubset;
    std::vector<std::string> outputFieldsSubset;
    std::vector<std::string> writeExtraSubset;

    EXPECT_EQ(sphexa::idTaggingOutputParameterRetrieval(settings, initCond, outputFileSuffix, 
        outFileSubset, writeFreqStrSubset, outputFieldsSubset, writeExtraSubset), true);
    EXPECT_EQ(writeFreqStrSubset, "14.83609");
}

TEST(IO, idTaggingWriteFrequencyRetrievalNegative)
{
    sphexa::InitSettings settings;
    settings["w_subset"] = -3.5;
    const std::string initCond = "dummy_cond";
    const std::string outputFileSuffix = ".dummy_suffix";
    std::string outFileSubset;
    std::string writeFreqStrSubset;
    std::vector<std::string> outputFieldsSubset;
    std::vector<std::string> writeExtraSubset;

    EXPECT_EQ(sphexa::idTaggingOutputParameterRetrieval(settings, initCond, outputFileSuffix, 
        outFileSubset, writeFreqStrSubset, outputFieldsSubset, writeExtraSubset), false);
}

TEST(IO, idTaggingOutputFileNaming)
{
    sphexa::InitSettings settings;
    settings["w_subset"] = 10;
    settings["o_subset"] = sphexa::Param(sphexa::stringToVectorValue("subset_output_file"), true);
    const std::string initCond = "dummy_cond";
    const std::string outputFileSuffix = ".dummy_suffix";
    std::string outFileSubset;
    std::string writeFreqStrSubset;
    std::vector<std::string> outputFieldsSubset;
    std::vector<std::string> writeExtraSubset;

    EXPECT_EQ(sphexa::idTaggingOutputParameterRetrieval(settings, initCond, outputFileSuffix, 
        outFileSubset, writeFreqStrSubset, outputFieldsSubset, writeExtraSubset), true);

    EXPECT_EQ(outFileSubset, "subset_output_file" + outputFileSuffix); 
}

TEST(IO, idTaggingOutputFileNamingDefault)
{
    sphexa::InitSettings settings;
    settings["w_subset"] = 10;
    const std::string initCond = "dummy_cond";
    const std::string outputFileSuffix = ".dummy_suffix";
    std::string outFileSubset;
    std::string writeFreqStrSubset;
    std::vector<std::string> outputFieldsSubset;
    std::vector<std::string> writeExtraSubset;

    EXPECT_EQ(sphexa::idTaggingOutputParameterRetrieval(settings, initCond, outputFileSuffix, 
        outFileSubset, writeFreqStrSubset, outputFieldsSubset, writeExtraSubset), true);

    EXPECT_EQ(outFileSubset, "dump_subset_" + initCond+outputFileSuffix); 
}

TEST(IO, idTaggingFieldsRetrieval)
{
    sphexa::InitSettings settings;
    settings["w_subset"] = 10;
    settings["f_subset"] = sphexa::Param(sphexa::stringToVectorValue("x,y,z,rho"), true);
    const std::string initCond = "dummy_cond";
    const std::string outputFileSuffix = ".dummy_suffix";
    std::string outFileSubset;
    std::string writeFreqStrSubset;
    std::vector<std::string> outputFieldsSubset;
    std::vector<std::string> writeExtraSubset;

    EXPECT_EQ(sphexa::idTaggingOutputParameterRetrieval(settings, initCond, outputFileSuffix, 
        outFileSubset, writeFreqStrSubset, outputFieldsSubset, writeExtraSubset), true);

    std::vector<std::string> expectedFields{"x", "y", "z", "rho"};
    EXPECT_EQ(outputFieldsSubset, expectedFields);
}

TEST(IO, idTaggingWriteExtraFrequencyRetrieval)
{
    sphexa::InitSettings settings;
    settings["w_subset"] = 10;
    settings["wextra_subset"] = sphexa::VectorValue{1, 10, 0.77};
    const std::string initCond = "dummy_cond";
    const std::string outputFileSuffix = ".dummy_suffix";
    std::string outFileSubset;
    std::string writeFreqStrSubset;
    std::vector<std::string> outputFieldsSubset;
    std::vector<std::string> writeExtraSubset;

    EXPECT_EQ(sphexa::idTaggingOutputParameterRetrieval(settings, initCond, outputFileSuffix, 
        outFileSubset, writeFreqStrSubset, outputFieldsSubset, writeExtraSubset), true);

    std::vector<std::string> expectedWriteExtra{"1", "10", "0.77"};
    EXPECT_EQ(writeExtraSubset, expectedWriteExtra);
}
