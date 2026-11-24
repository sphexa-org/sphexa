/*
 * MIT License
 *
 * Copyright (c) 2026 CSCS, ETH Zurich
 *               2026 University of Basel
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

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "io/ifile_io_impl.h"

using namespace sphexa;

namespace
{

void runRoundTrip(bool bp5, const std::string& path)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    if (rank == 0) { std::filesystem::remove_all(path); }
    MPI_Barrier(MPI_COMM_WORLD);

    size_t localCount   = rank == 0 ? 2 : 0;
    size_t globalOffset = 0;
    MPI_Exscan(&localCount, &globalOffset, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) { globalOffset = 0; }
    size_t globalCount = localCount;
    MPI_Allreduce(MPI_IN_PLACE, &globalCount, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

    {
        auto writer = makeOpenPMDWriter(MPI_COMM_WORLD, bp5, 16, "all", "{}", "{}");

        writer->addStep(0, 0, path);
        double setting = 42.0;
        writer->fileAttribute("setting", &setting, 1);
        writer->fileAttribute("backend", std::string(bp5 ? "bp5" : "hdf5"));
        writer->closeStep();

        std::vector<double>   position(localCount + 2, -1.0);
        std::vector<uint64_t> ids(localCount + 2, std::numeric_limits<uint64_t>::max());
        for (size_t i = 0; i < localCount; ++i)
        {
            position[i + 1] = double(globalOffset + i) + 0.25;
            ids[i + 1]      = globalOffset + i;
        }

        writer->addStep(1, localCount + 1, path);
        uint64_t iteration = rank == 0 ? 17 : 99;
        writer->stepAttribute("iteration", &iteration, 1);
        double time = rank == 0 ? 1.25 : 9.0;
        writer->stepAttribute("time", &time, 1);
        writer->writeField("x", position.data(), 0);
        writer->writeField("id", ids.data(), 1);
        writer->closeStep();
    }

    {
        auto reader = makeOpenPMDReader(MPI_COMM_WORLD, bp5);
        reader->setStep(path, 0, FileMode::collective);

        double setting = 0.0;
        reader->fileAttribute("setting", &setting, 1);
        EXPECT_DOUBLE_EQ(setting, 42.0);
        std::string backend;
        reader->fileAttribute("backend", backend);
        EXPECT_EQ(backend, bp5 ? "bp5" : "hdf5");

        uint64_t iteration = 0;
        reader->stepAttribute("iteration", &iteration, 1);
        EXPECT_EQ(iteration, 17);
        double time = 0.0;
        reader->stepAttribute("time", &time, 1);
        EXPECT_DOUBLE_EQ(time, 1.25);

        EXPECT_EQ(reader->globalNumParticles(), globalCount);
        size_t                readFirst = globalCount * size_t(rank) / size_t(numRanks);
        size_t                readLast  = globalCount * size_t(rank + 1) / size_t(numRanks);
        std::vector<double>   position(readLast - readFirst);
        std::vector<uint64_t> ids(readLast - readFirst);
        reader->readField("x", position.data());
        reader->readField("id", ids.data());
        for (size_t i = readFirst; i < readLast; ++i)
        {
            EXPECT_DOUBLE_EQ(position[i - readFirst], double(i) + 0.25);
            EXPECT_EQ(ids[i - readFirst], i);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace

TEST(OpenPMDIO, Hdf5RoundTrip) { runRoundTrip(false, "openpmd_roundtrip.h5"); }

TEST(OpenPMDIO, Bp5RoundTrip) { runRoundTrip(true, "openpmd_roundtrip.bp5"); }
