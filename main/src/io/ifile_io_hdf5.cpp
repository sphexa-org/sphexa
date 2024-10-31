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
 * @brief File I/O interface implementation with H5Part
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <mpi.h>

#include <filesystem>
#include <string>
#include <variant>
#include <vector>

#include "ifile_io_impl.h"

#ifdef SPH_EXA_HAVE_H5HUT
#include "h5part_wrapper.hpp"
#endif

namespace sphexa
{

#ifdef SPH_EXA_HAVE_H5HUT

class H5PartWriter final : public IFileWriter
{
public:
    using Base      = IFileWriter;
    using FieldType = typename Base::FieldType;

    explicit H5PartWriter(MPI_Comm comm)
        : comm_(comm)
    {
        MPI_Comm_rank(comm, &rank_);
        MPI_Comm_size(comm, &numRanks_);
    }

    ~H5PartWriter() override { closeStep(); }

    [[nodiscard]] int rank() const override { return rank_; }
    [[nodiscard]] int numRanks() const override { return numRanks_; }

    std::string suffix() const override { return ".h5"; }

    void addStep(size_t firstIndex, size_t lastIndex, std::string path) override
    {
        firstIndex_ = firstIndex;

        if (!h5File_ || path != pathStep_)
        {
            closeStep();
            h5File_ = fileutils::openH5Part(path, H5_O_RDWR, comm_);
        }

        if (lastIndex > firstIndex)
        {
            // create the next step
            h5_int64_t numSteps = H5GetNumSteps(h5File_);
            H5SetStep(h5File_, numSteps);

            uint64_t numParticles = lastIndex - firstIndex;
            // set number of particles that each rank will write
            H5PartSetNumParticles(h5File_, numParticles);
        }
        pathStep_ = path;
    }

    void stepAttribute(const std::string& key, FieldType val, int64_t size) override
    {
        std::visit(
            [this, &key, size](auto arg) { //
                fileutils::H5WriteStepAttribT(h5File_, key.c_str(), arg, size);
            },
            val);
    }

    void fileAttribute(const std::string& key, FieldType val, int64_t size) override
    {
        std::visit(
            [this, &key, size](auto arg) { //
                fileutils::H5WriteFileAttribT(h5File_, key.c_str(), arg, size);
            },
            val);
    }

    void writeField(const std::string& key, FieldType field, int = 0) override
    {
        std::visit([this, &key](auto arg) { fileutils::H5PartWriteDataT(h5File_, key.c_str(), arg + firstIndex_); },
                   field);
    }

    void closeStep() override
    {
        if (h5File_)
        {
            H5CloseFile(h5File_);
            h5File_ = 0;
        }
    }

private:
    int      rank_{0}, numRanks_{0};
    MPI_Comm comm_;

    size_t      firstIndex_{0};
    std::string pathStep_;

    h5_file_t h5File_{0};
};

std::unique_ptr<IFileWriter> makeH5PartWriter(MPI_Comm comm) { return std::make_unique<H5PartWriter>(comm); }

inline auto partitionRange(size_t R, size_t i, size_t N)
{
    size_t s = R / N;
    size_t r = R % N;
    if (i < r)
    {
        size_t start = (s + 1) * i;
        size_t end   = start + s + 1;
        return std::make_tuple(start, end);
    }
    else
    {
        size_t start = (s + 1) * r + s * (i - r);
        size_t end   = start + s;
        return std::make_tuple(start, end);
    }
}

class H5PartReader final : public IFileReader
{
public:
    using Base      = IFileReader;
    using FieldType = typename Base::FieldType;

    explicit H5PartReader(MPI_Comm comm)
        : comm_(comm)
        , h5File_{0}
    {
        MPI_Comm_rank(comm, &rank_);
    }

    ~H5PartReader() override { closeStep(); }

    [[nodiscard]] int     rank() const override { return rank_; }
    [[nodiscard]] int64_t numParticles() const override
    {
        if (!h5File_) { throw std::runtime_error("Cannot get number of particles: file not open\n"); }
        return H5PartGetNumParticles(h5File_);
    }

    /*! @brief open a file at a given step
     *
     * @param path  filesystem path
     * @param step  snapshot index to load from
     * @param mode  collective mode causes MPI ranks to distribute particles amongst themselves,
     *              independent mode causes all MPI ranks to load all particles
     */
    void setStep(std::string path, int step, FileMode mode) override
    {
        closeStep();
        pathStep_ = path;
        h5File_   = fileutils::openH5Part(path, H5_O_RDWR, comm_);

        if (H5GetNumSteps(h5File_) == 0) { return; }

        // set step to last step in file if negative
        if (step < 0) { step = H5GetNumSteps(h5File_) - 1; }
        H5SetStep(h5File_, step);

        globalCount_ = H5PartGetNumParticles(h5File_);
        if (globalCount_ < 1) { return; }

        int rank, numRanks;
        MPI_Comm_rank(comm_, &rank);
        MPI_Comm_size(comm_, &numRanks);

        if (mode == FileMode::collective)
        {
            std::tie(firstIndex_, lastIndex_) = partitionRange(globalCount_, rank, numRanks);
            localCount_                       = lastIndex_ - firstIndex_;
            H5PartSetView(h5File_, firstIndex_, lastIndex_ - 1);
        }
        else { std::tie(firstIndex_, lastIndex_, localCount_) = std::make_tuple(0, globalCount_, globalCount_); }
    }

    std::vector<std::string> fileAttributes() override
    {
        if (h5File_) { return fileutils::fileAttributeNames(h5File_); }
        else { throw std::runtime_error("Cannot read file attributes: file not opened\n"); }
    }

    std::vector<std::string> stepAttributes() override
    {
        if (h5File_) { return fileutils::stepAttributeNames(h5File_); }
        else { throw std::runtime_error("Cannot read file attributes: file not opened\n"); }
    }

    int64_t fileAttributeSize(const std::string& key) override
    {
        int64_t   attrIndex = fileAttributeIndex(key);
        h5_size_t attrSize;
        int64_t   typeId;
        char      dummy[256];
        H5GetFileAttribInfo(h5File_, attrIndex, dummy, 256, &typeId, &attrSize);
        return attrSize;
    }

    int64_t stepAttributeSize(const std::string& key) override
    {
        int64_t   attrIndex = stepAttributeIndex(key);
        h5_size_t attrSize;
        int64_t   typeId;
        char      dummy[256];
        H5GetStepAttribInfo(h5File_, attrIndex, dummy, 256, &typeId, &attrSize);
        return attrSize;
    }

    void fileAttribute(const std::string& key, FieldType val, int64_t size) override
    {
        std::visit(
            [this, size, &key](auto arg)
            {
                auto index = fileAttributeIndex(key);
                fileutils::readH5PartFileAttribute(arg, size, index, h5File_);
            },
            val);
    }

    void stepAttribute(const std::string& key, FieldType val, int64_t size) override
    {
        std::visit(
            [this, size, &key](auto arg)
            {
                auto index = stepAttributeIndex(key);
                fileutils::readH5PartStepAttribute(arg, size, index, h5File_);
            },
            val);
    }

    void readField(const std::string& key, FieldType field) override
    {
        auto err =
            std::visit([this, &key](auto arg) { return fileutils::H5PartReadDataT(h5File_, key.c_str(), arg); }, field);
        if (err != H5_SUCCESS) { throw std::runtime_error("Could not read field: " + key); }
    }

    uint64_t localNumParticles() override { return localCount_; }

    uint64_t globalNumParticles() override { return globalCount_; }

    void closeStep() override
    {
        if (h5File_)
        {
            H5CloseFile(h5File_);
            h5File_ = 0;
        }
    }

private:
    int64_t stepAttributeIndex(const std::string& key)
    {
        auto    attributes = fileutils::stepAttributeNames(h5File_);
        int64_t attrIndex  = std::find(attributes.begin(), attributes.end(), key) - attributes.begin();
        if (attrIndex == attributes.size()) { throw std::out_of_range("Attribute " + key + " does not exist\n"); }
        return attrIndex;
    }

    int64_t fileAttributeIndex(const std::string& key)
    {
        auto    attributes = fileutils::fileAttributeNames(h5File_);
        int64_t attrIndex  = std::find(attributes.begin(), attributes.end(), key) - attributes.begin();
        if (attrIndex == attributes.size()) { throw std::out_of_range("Attribute " + key + " does not exist\n"); }
        return attrIndex;
    }

    int      rank_{0};
    MPI_Comm comm_;

    int64_t     firstIndex_, lastIndex_;
    int64_t     localCount_;
    int64_t     globalCount_;
    std::string pathStep_;

    h5_file_t h5File_;
};

std::unique_ptr<IFileReader> makeH5PartReader(MPI_Comm comm) { return std::make_unique<H5PartReader>(comm); }

#else

std::unique_ptr<IFileWriter> makeH5PartWriter(MPI_Comm) { return {}; }
std::unique_ptr<IFileReader> makeH5PartReader(MPI_Comm) { return std::make_unique<UnimplementedReader>(); }

#endif

} // namespace sphexa
