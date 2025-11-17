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

TEST(IO, idTaggingWriteFrequencyRetrieval)
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

    settings["w_subset"] = 14.83609;

    EXPECT_EQ(sphexa::idTaggingOutputParameterRetrieval(settings, initCond, outputFileSuffix, 
        outFileSubset, writeFreqStrSubset, outputFieldsSubset, writeExtraSubset), true);

    EXPECT_EQ(writeFreqStrSubset, "14.83609");

    settings["w_subset"] = -3.5;
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

    settings.erase("o_subset");

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
