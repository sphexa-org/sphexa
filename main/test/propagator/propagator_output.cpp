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
 * @brief Unit tests for ipropagator functionalities
 *
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <numeric>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "io/ifile_io_impl.h"
#include "propagator_test.hpp"

TEST(IO, saveFieldsFull)
{
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    std::string testfile = "output_field.h5";
    if (rank == 0 && std::filesystem::exists(testfile)) { std::filesystem::remove(testfile); }
    MPI_Barrier(MPI_COMM_WORLD);

    size_t first = 0;
    size_t last = 50;
    std::vector<double> testField(50);
    std::iota(testField.begin(), testField.end(), 0.0);
    FieldVariant fieldData;
    fieldData = &testField;

    auto fileWriter = sphexa::makeH5PartWriter(MPI_COMM_WORLD);
    fileWriter->addStep(first, last, testfile);

    PropagatorTest<Domain, Dataset>::AccVector<double> accFieldData;
    std::vector<char> hostScratch;
    PropagatorTest<Domain, Dataset>::outputField(fileWriter.get(), first, last, fieldData, "testField", 0, accFieldData, hostScratch);

    fileWriter->closeStep();

    auto reader = sphexa::makeH5PartReader(MPI_COMM_WORLD);
    reader->setStep(testfile, 0, sphexa::FileMode::collective);

    std::vector<double> readField(reader->localNumParticles());
    reader->readField("testField", readField.data());

    EXPECT_EQ(testField, readField);

    reader->closeStep();

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST(IO, saveFieldsSubDomain)
{
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    std::string testfile = "output_field.h5";
    if (rank == 0 && std::filesystem::exists(testfile)) { std::filesystem::remove(testfile); }
    MPI_Barrier(MPI_COMM_WORLD);

    size_t first = 20;
    size_t last = 30;
    std::vector<double> testField(50);
    std::iota(testField.begin(), testField.end(), 0.0);
    FieldVariant fieldData;
    fieldData = &testField;

    auto fileWriter = sphexa::makeH5PartWriter(MPI_COMM_WORLD);
    fileWriter->addStep(first, last, testfile);

    PropagatorTest<Domain, Dataset>::AccVector<double> accFieldData;
    std::vector<char> hostScratch;
    PropagatorTest<Domain, Dataset>::outputField(fileWriter.get(), first, last, fieldData, "testField", 0, accFieldData, hostScratch);

    fileWriter->closeStep();

    auto reader = sphexa::makeH5PartReader(MPI_COMM_WORLD);
    reader->setStep(testfile, 0, sphexa::FileMode::collective);

    std::vector<double> readField(reader->localNumParticles());
    reader->readField("testField", readField.data());

    std::vector<double> expectedField(last - first);
    std::copy(testField.begin() + first, testField.begin() + last, expectedField.begin());
    EXPECT_EQ(expectedField, readField);

    reader->closeStep();

    MPI_Barrier(MPI_COMM_WORLD);
}