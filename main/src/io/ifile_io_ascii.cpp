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
 * @brief ASCII file I/O
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <mpi.h>

#include <filesystem>
#include <string>
#include <vector>

#include "cstone/primitives/gather.hpp"

#include "file_utils.hpp"
#include "ifile_io_impl.h"

namespace sphexa
{

class AsciiWriter : public IFileWriter
{
public:
    using Base      = IFileWriter;
    using FieldType = typename Base::FieldType;

    AsciiWriter(MPI_Comm comm)
        : comm_(comm)
    {
        MPI_Comm_rank(comm, &rank_);
        MPI_Comm_size(comm, &numRanks_);
    }

    [[nodiscard]] int rank() const override { return rank_; }
    [[nodiscard]] int numRanks() const override { return numRanks_; }

    std::string suffix() const override { return ""; }
    // TODO: better implementation needed, can we remove settingsWriting flag?
    void addStep(size_t firstIndex, size_t lastIndex, std::string path, bool settingsWriting = false) override
    {
        firstIndexStep_ = firstIndex;
        lastIndexStep_  = lastIndex;
        pathStep_       = path;
    }

    void stepAttribute(const std::string& key, FieldType val, int64_t /*size*/) override
    {
        if (key == "iteration") { iterationStep_ = *std::get<const uint64_t*>(val); }
    }

    void fileAttribute(const std::string& key, FieldType val, int64_t /*size*/) override {}

    void writeField(const std::string& /*key*/, FieldType field, int col) override
    {
        columns_.push_back(col);
        std::visit(
            [this](auto arg)
            {
                std::vector<std::decay_t<decltype(*arg)>> vec(lastIndexStep_ - firstIndexStep_);
                std::copy_n(arg + firstIndexStep_, vec.size(), vec.data());
                stepBuffer_.push_back(std::move(vec));
            },
            field);
    }

    void closeStep() override
    {
        if (lastIndexStep_ == firstIndexStep_) { return; }

        const char separator = ' ';
        pathStep_ += "." + std::to_string(iterationStep_) + ".txt";

        int rank, numRanks;
        MPI_Comm_rank(comm_, &rank);
        MPI_Comm_size(comm_, &numRanks);

        std::vector<FieldType> fieldPointers;
        for (const auto& v : stepBuffer_)
        {
            std::visit([&fieldPointers](const auto& arg) { fieldPointers.push_back(arg.data()); }, v);
        }

        cstone::sort_by_key(columns_.begin(), columns_.end(), fieldPointers.begin());

        for (int turn = 0; turn < numRanks; turn++)
        {
            if (turn == rank)
            {
                try
                {
                    bool append = rank != 0;
                    fileutils::writeAscii(0, lastIndexStep_ - firstIndexStep_, pathStep_, append, fieldPointers,
                                          separator);
                }
                catch (std::runtime_error& ex)
                {
                    if (rank == 0) fprintf(stderr, "ERROR: %s Terminating\n", ex.what());
                    MPI_Abort(comm_, 1);
                }
            }

            MPI_Barrier(comm_);
        }
        columns_.clear();
        stepBuffer_.clear();
    }

private:
    int                            rank_{0}, numRanks_{0};
    MPI_Comm                       comm_;
    int64_t                        firstIndexStep_{0}, lastIndexStep_{0};
    std::string                    pathStep_;
    std::vector<int>               columns_;
    std::vector<Base::FieldVector> stepBuffer_;
    uint64_t                       iterationStep_{0};
};

std::unique_ptr<IFileWriter> makeAsciiWriter(MPI_Comm comm) { return std::make_unique<AsciiWriter>(comm); }

} // namespace sphexa
