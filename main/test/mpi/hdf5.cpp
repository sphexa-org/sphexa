/*
 * MIT License
 *
 * Copyright (c) 2023 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * @brief Unit tests for I/O related functionality
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <iostream>
#include <numeric>

#include "gtest/gtest.h"

#include "sph/hydro_turb/turbulence_data.hpp"
#include "sph/particles_data.hpp"
#include "io/ifile_io_hdf5.hpp"

using namespace sphexa;

TEST(HDF5IO, stepAttribute)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string testfile = "step_attributes.h5";
    if (rank == 0 && std::filesystem::exists(testfile)) { std::filesystem::remove(testfile); }
    MPI_Barrier(MPI_COMM_WORLD);

    {
        H5PartWriter writer(MPI_COMM_WORLD);
        writer.addStep(0, 10, testfile);

        double float64Attr = 0.5;
        writer.stepAttribute("float64Attr", &float64Attr, 1);
        int64_t int64Attr = 42;
        writer.stepAttribute("int64Attr", &int64Attr, 1);
        uint64_t uint64Attr = uint64_t(2) << 40;
        writer.stepAttribute("uint64Attr", &uint64Attr, 1);
        char int8Attr = 1;
        writer.stepAttribute("int8Attr", &int8Attr, 1);
        writer.closeStep();
    }
    {
        H5PartReader reader(MPI_COMM_WORLD);
        double       float64Attr;
        int64_t      int64Attr;
        uint64_t     uint64Attr;
        char         int8Attr;
        reader.setStep(testfile, 0);
        reader.stepAttribute("float64Attr", &float64Attr, 1);
        reader.stepAttribute("int64Attr", &int64Attr, 1);
        reader.stepAttribute("uint64Attr", &uint64Attr, 1);
        reader.stepAttribute("int8Attr", &int8Attr, 1);

        // providing a wrong type should produce a runtime exception, HDF5 does not do conversions for attributes
        int ttotInt;
        EXPECT_THROW(reader.stepAttribute("float64Attr", &ttotInt, 1), std::runtime_error);

        EXPECT_EQ(float64Attr, 0.5);
        EXPECT_EQ(int64Attr, 42);
        EXPECT_EQ(uint64Attr, uint64_t(2) << 40);
        EXPECT_EQ(int8Attr, 1);
        reader.closeStep();
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST(HDF5IO, fields)
{
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    std::string testfile = "fields.h5";
    if (rank == 0 && std::filesystem::exists(testfile)) { std::filesystem::remove(testfile); }
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<double> x(10);
    std::vector<int>    nc(10);
    std::iota(x.begin(), x.end(), rank * x.size());
    std::iota(nc.begin(), nc.end(), rank * nc.size());

    size_t first              = 1;
    size_t last               = first + x.size();
    size_t localSizeWithHalos = first + x.size() + 1;

    {
        std::vector<double> xWithHalos(localSizeWithHalos);
        std::vector<int>    ncWithHalos(localSizeWithHalos);
        std::copy(x.begin(), x.end(), xWithHalos.begin() + first);
        std::copy(nc.begin(), nc.end(), ncWithHalos.begin() + first);

        H5PartWriter writer(MPI_COMM_WORLD);
        writer.addStep(first, last, testfile);

        writer.writeField("x", xWithHalos.data());
        writer.writeField("nc", ncWithHalos.data());

        writer.closeStep();
    }
    {
        H5PartReader reader(MPI_COMM_WORLD);
        reader.setStep(testfile, 0);

        EXPECT_EQ(reader.localNumParticles(), 10);
        EXPECT_EQ(reader.globalNumParticles(), 10 * numRanks);

        // Note: HDF5 will do data-type conversion between float and double and int32 and int64
        // if the type in file does not match the type in memory
        std::vector<double> xread(reader.localNumParticles());
        std::vector<int>    ncread(reader.localNumParticles());

        reader.readField("x", xread.data());
        reader.readField("nc", ncread.data());

        EXPECT_EQ(x, xread);
        EXPECT_EQ(nc, ncread);

        reader.closeStep();
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST(HDF5IO, particleData)
{
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    std::string testfile = "pdata.h5";
    if (rank == 0 && std::filesystem::exists(testfile)) { std::filesystem::remove(testfile); }
    MPI_Barrier(MPI_COMM_WORLD);

    using Dataset = sphexa::ParticlesData<double, uint64_t, cstone::CpuTag>;

    {
        Dataset data;
        data.iteration = 42;
        data.ttot      = 3.14159;
        data.ngmax     = 1000;
        H5PartWriter writer(MPI_COMM_WORLD);
        writer.addStep(0, 1, testfile);
        data.loadOrStoreAttributes(&writer);
        writer.closeStep();
    }
    {
        Dataset      data;
        H5PartReader reader(MPI_COMM_WORLD);
        reader.setStep(testfile, 0);
        data.loadOrStoreAttributes(&reader);
        EXPECT_EQ(data.iteration, 42);
        EXPECT_EQ(data.ttot, 3.14159);
        EXPECT_EQ(data.ngmax, 1000);
        reader.closeStep();
    }
}

TEST(HDF5IO, box)
{
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    std::string testfile = "box.h5";
    if (rank == 0 && std::filesystem::exists(testfile)) { std::filesystem::remove(testfile); }
    MPI_Barrier(MPI_COMM_WORLD);

    using Box = cstone::Box<double>;
    auto open = cstone::BoundaryType::open;
    auto pbc  = cstone::BoundaryType::periodic;

    {
        Box          box(0, 1, 0, 2, 0, 3, pbc, open, pbc);
        H5PartWriter writer(MPI_COMM_WORLD);
        writer.addStep(0, 1, testfile);
        box.loadOrStore(&writer);
        writer.closeStep();
    }
    {
        Box          box(0, 1, 0, 1, 0, 1, open, pbc, open);
        H5PartReader reader(MPI_COMM_WORLD);
        reader.setStep(testfile, 0);
        box.loadOrStore(&reader);
        EXPECT_EQ(box.xmin(), 0.0);
        EXPECT_EQ(box.xmax(), 1.0);
        EXPECT_EQ(box.ymin(), 0.0);
        EXPECT_EQ(box.ymax(), 2.0);
        EXPECT_EQ(box.zmin(), 0.0);
        EXPECT_EQ(box.zmax(), 3.0);

        EXPECT_EQ(box.lx(), 1.0);
        EXPECT_EQ(box.ly(), 2.0);
        EXPECT_EQ(box.lz(), 3.0);

        EXPECT_NEAR(box.ilx(), 1. / 1.0, 1e-6);
        EXPECT_NEAR(box.ily(), 1. / 2.0, 1e-6);
        EXPECT_NEAR(box.ilz(), 1. / 3.0, 1e-6);

        EXPECT_EQ(box.boundaryX(), pbc);
        EXPECT_EQ(box.boundaryY(), open);
        EXPECT_EQ(box.boundaryZ(), pbc);

        reader.closeStep();
    }
}

TEST(HDF5IO, turbulenceData)
{
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    std::string testfile = "turbulence.h5";
    if (rank == 0 && std::filesystem::exists(testfile)) { std::filesystem::remove(testfile); }
    MPI_Barrier(MPI_COMM_WORLD);

    using Data = sph::TurbulenceData<double, cstone::CpuTag>;

    std::map<std::string, double> constants{
        {"solWeight", 0.5},      {"stMaxModes", 100000}, {"Lbox", 1.0},       {"stMachVelocity", 0.3e0},
        {"firstTimeStep", 1e-4}, {"epsilon", 1e-15},     {"rngSeed", 251299}, {"stSpectForm", 1},
        {"mTotal", 1.0},         {"powerLawExp", 5 / 3}, {"anglesExp", 2.0},  {"gamma", 1.001},
        {"mui", 0.62},           {"u0", 1000.},          {"Kcour", 0.4}};

    Data data(constants, false);
    std::iota(data.phases.begin(), data.phases.end(), 0.0);

    std::uniform_int_distribution<int> dist;
    for (int i = 0; i < 2500; ++i)
    {
        dist(data.gen);
    }

    {
        H5PartWriter writer(MPI_COMM_WORLD);
        writer.addStep(0, 1, testfile);
        data.loadOrStore(&writer);
        writer.closeStep();
    }
    {
        Data probe(constants, false);

        EXPECT_NE(probe.gen, data.gen);

        H5PartReader reader(MPI_COMM_WORLD);
        reader.setStep(testfile, 0);
        probe.loadOrStore(&reader);

        EXPECT_EQ(probe.gen, data.gen);
        EXPECT_EQ(probe.phases, data.phases);

        reader.closeStep();
    }
}
