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

/*! @file
 * @brief File I/O interface implementation with openPMD
 */

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include <mpi.h>

#include "ifile_io_impl.h"

#ifdef SPH_EXA_HAVE_OPENPMD

#include <openPMD/auxiliary/JSON.hpp>
#include <openPMD/openPMD.hpp>

#include "cstone/primitives/mpi_wrappers.hpp"

namespace sphexa
{

namespace
{

constexpr const char* particleSpecies = "particles";

bool collectiveData(const std::string& mode) { return mode == "all" || mode == "data"; }

std::string backendConfiguration(bool bp5, const std::string& collective, const std::string& input,
                                 const std::string& engineOverrides, bool reading)
{
    const std::string required =
        bp5 ? std::string(
                  R"({"backend":"adios2","iteration_encoding":"file_based","adios2":{"engine":{"type":"bp5","preferred_flush_target":"disk","access_mode":")") +
                  (reading ? "readrandomaccess" : "write") + R"(","parameters":{"asyncwrite":"off"}}}})"
            : R"({"backend":"hdf5","iteration_encoding":"group_based"})";

    std::string collectiveConfig = "{}";
    if (bp5)
    {
        if (!collectiveData(collective))
        {
            collectiveConfig = R"({"adios2":{"engine":{"parameters":{"aggregationtype":"everyonewrites"}}}})";
        }
    }
    else
    {
        collectiveConfig =
            std::string(R"({"hdf5":{"independent_stores":)") + (collectiveData(collective) ? "false" : "true") + "}}";
    }

    const auto requestedConfig = openPMD::json::merge(input, collectiveConfig);
    const auto canonical       = openPMD::json::merge(requestedConfig, engineOverrides);
    auto       folded          = canonical;
    for (auto& c : folded)
    {
        if (c >= 'A' && c <= 'Z') { c += 'a' - 'A'; }
    }

    const auto requested = openPMD::json::merge(required, folded);
    if (openPMD::json::merge(requested, required) != requested)
    {
        throw std::runtime_error("OpenPMD configuration conflicts with the selected backend, iteration encoding, "
                                 "access mode or synchronous disk output");
    }
    return openPMD::json::merge(required, canonical);
}

std::string seriesPath(const std::string& path, bool bp5)
{
    if (!bp5) { return path; }
    return (std::filesystem::path(path) / "data_%06T.bp5").string();
}

auto createOffsetRecord(openPMD::Iteration& iteration)
{
    auto& record = iteration.particles[particleSpecies]["positionOffset"];
    record.setUnitDimension({1, 0, 0, 0, 0, 0, 0});
    return record;
}

template<class T>
void setPositionOffset(openPMD::Iteration& iteration, uint64_t globalCount)
{
    auto record = createOffsetRecord(iteration);
    for (const char* componentName : {"x", "y", "z"})
    {
        auto component = record[componentName];
        component.resetDataset(openPMD::Dataset(openPMD::determineDatatype<T>(), {globalCount}));
        component.setUnitSI(1.0);
        component.makeConstant(T{0});
    }
}

template<class T>
void writeAttribute(openPMD::Attributable& attributable, const std::string& key, const T* values, int64_t size)
{
    if (size < 1) { throw std::runtime_error("Attribute '" + key + "' has no values"); }
    if (size == 1) { attributable.setAttribute(key, values[0]); }
    else { attributable.setAttribute(key, std::vector<T>(values, values + size)); }
}

template<class T>
void readAttribute(openPMD::Attributable& attributable, const std::string& key, T* values, int64_t size)
{
    const auto attribute = attributable.getAttribute(key);
    if (auto vectorValue = attribute.template getOptional<std::vector<T>>(); vectorValue)
    {
        if (int64_t(vectorValue->size()) != size)
        {
            throw std::runtime_error("Attribute '" + key + "' has size " + std::to_string(vectorValue->size()) +
                                     ", expected " + std::to_string(size));
        }
        std::copy(vectorValue->begin(), vectorValue->end(), values);
        return;
    }
    if (size != 1) { throw std::runtime_error("Attribute '" + key + "' is scalar, expected size > 1"); }
    values[0] = attribute.template get<T>();
}

int64_t attributeSize(const openPMD::Attribute& attribute)
{
    if (auto value = attribute.template getOptional<std::string>(); value)
    {
        return std::max<int64_t>(2, int64_t(value->size()) + 1);
    }
    auto check = [&attribute]<class T>() -> std::optional<size_t>
    {
        auto value = attribute.template getOptional<std::vector<T>>();
        return value ? std::optional<size_t>{value->size()} : std::nullopt;
    };

    if (auto size = check.template operator()<double>(); size) { return size.value(); }
    if (auto size = check.template operator()<float>(); size) { return size.value(); }
    if (auto size = check.template operator()<char>(); size) { return size.value(); }
    if (auto size = check.template operator()<unsigned char>(); size) { return size.value(); }
    if (auto size = check.template operator()<signed char>(); size) { return size.value(); }
    if (auto size = check.template operator()<short>(); size) { return size.value(); }
    if (auto size = check.template operator()<int>(); size) { return size.value(); }
    if (auto size = check.template operator()<long>(); size) { return size.value(); }
    if (auto size = check.template operator()<long long>(); size) { return size.value(); }
    if (auto size = check.template operator()<unsigned short>(); size) { return size.value(); }
    if (auto size = check.template operator()<unsigned>(); size) { return size.value(); }
    if (auto size = check.template operator()<unsigned long>(); size) { return size.value(); }
    if (auto size = check.template operator()<unsigned long long>(); size) { return size.value(); }
    if (auto size = check.template operator()<std::string>(); size) { return size.value(); }
    return 1;
}

} // namespace

class OpenPMDWriter final : public IFileWriter
{
public:
    using Base      = IFileWriter;
    using FieldType = typename Base::FieldType;
    using PendingAttribute =
        std::variant<std::vector<double>, std::vector<float>, std::vector<char>, std::vector<unsigned char>,
                     std::vector<signed char>, std::vector<short>, std::vector<int>, std::vector<long>,
                     std::vector<long long>, std::vector<unsigned short>, std::vector<unsigned>,
                     std::vector<unsigned long>, std::vector<unsigned long long>>;

    OpenPMDWriter(MPI_Comm comm, bool bp5, size_t stagingBytes, std::string collective, std::string config,
                  std::string engineConfig)
        : comm_(comm)
        , bp5_(bp5)
        , stagingBytes_(std::max<size_t>(1, stagingBytes))
        , collective_(std::move(collective))
        , config_(std::move(config))
        , engineConfig_(std::move(engineConfig))
    {
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &numRanks_);
    }

    ~OpenPMDWriter() override { closeStep(); }

    int         rank() const override { return rank_; }
    int         numRanks() const override { return numRanks_; }
    std::string suffix() const override { return bp5_ ? ".bp5" : ".h5"; }

    void addStep(size_t firstIndex, size_t lastIndex, std::string path) override
    {
        if (activeIteration_) { throw std::runtime_error("Cannot add a checkpoint before closing the previous one"); }

        firstIndex_  = firstIndex;
        localCount_  = lastIndex - firstIndex;
        globalCount_ = localCount_;
        MPI_Allreduce(MPI_IN_PLACE, &globalCount_, 1, MPI_UINT64_T, MPI_SUM, comm_);
        if (bp5_ && globalCount_ == 0)
        {
            pathStep_ = std::move(path);
            return;
        }

        openSeries(path);
        pathStep_ = path;

        if (globalCount_ == 0) { return; }

        uint64_t offset = 0;
        MPI_Exscan(&localCount_, &offset, 1, MPI_UINT64_T, MPI_SUM, comm_);
        if (rank_ == 0) { offset = 0; }
        globalOffset_ = offset;

        iterationIndex_        = nextIteration_[path]++;
        activeIteration_       = true;
        positionOffsetWritten_ = false;
        auto iteration         = series_->iterations[iterationIndex_];
        iteration.open();
        iteration.setTimeUnitSI(1.0);
        iteration.setAttribute("sphexa_schema", uint32_t{1});
    }

    void stepAttribute(const std::string& key, FieldType values, int64_t size) override
    {
        requireIteration(key);
        auto iteration = series_->iterations[iterationIndex_];
        std::visit(
            [&](auto pointer)
            {
                using T = std::remove_cv_t<std::remove_pointer_t<decltype(pointer)>>;
                std::vector<T> rankZeroValue(size);
                if (size > 0 && rank_ == 0) { std::copy(pointer, pointer + size, rankZeroValue.begin()); }
                MPI_Bcast(rankZeroValue.data(), size, MpiType<T>{}, 0, comm_);
                writeAttribute(iteration, key, rankZeroValue.data(), size);
                if (key == "time" && size == 1) { iteration.setTime(double(rankZeroValue[0])); }
                if (key == "minDt" && size == 1) { iteration.setDt(double(rankZeroValue[0])); }
            },
            values);
    }

    void stepAttribute(const std::string& key, const std::string& value) override
    {
        requireIteration(key);
        int length = rank_ == 0 ? int(value.size()) : 0;
        MPI_Bcast(&length, 1, MPI_INT, 0, comm_);
        std::string rankZeroValue(length, '\0');
        if (rank_ == 0) { rankZeroValue = value; }
        MPI_Bcast(rankZeroValue.data(), length, MPI_CHAR, 0, comm_);
        series_->iterations[iterationIndex_].setAttribute(key, rankZeroValue);
    }

    void fileAttribute(const std::string& key, FieldType values, int64_t size) override
    {
        if (bp5_ && !series_)
        {
            std::visit(
                [&](auto pointer)
                {
                    using T                     = std::remove_cv_t<std::remove_pointer_t<decltype(pointer)>>;
                    pendingFileAttributes_[key] = std::vector<T>(pointer, pointer + size);
                },
                values);
            return;
        }
        requireSeries(key);
        std::visit([&](auto pointer) { writeAttribute(*series_, key, pointer, size); }, values);
    }

    void fileAttribute(const std::string& key, const std::string& value) override
    {
        if (bp5_ && !series_)
        {
            pendingStringAttributes_[key] = value;
            return;
        }
        requireSeries(key);
        series_->setAttribute(key, value);
    }

    void writeField(const std::string& key, FieldType field, int = 0) override
    {
        requireIteration(key);
        auto iteration = series_->iterations[iterationIndex_];
        auto species   = iteration.particles[particleSpecies];

        std::visit(
            [&](const auto* source)
            {
                using T    = std::remove_cv_t<std::remove_pointer_t<decltype(source)>>;
                auto store = [&](auto component)
                {
                    component.resetDataset(openPMD::Dataset(openPMD::determineDatatype<T>(), {globalCount_}));
                    component.setUnitSI(1.0);

                    uint64_t capacity    = std::max<uint64_t>(1, stagingBytes_ / sizeof(T));
                    uint64_t localChunks = (localCount_ + capacity - 1) / capacity;
                    uint64_t chunkRounds = 0;
                    MPI_Allreduce(&localChunks, &chunkRounds, 1, MPI_UINT64_T, MPI_MAX, comm_);
                    T emptyValue{};
                    for (uint64_t chunk = 0; chunk < chunkRounds; ++chunk)
                    {
                        uint64_t begin = std::min<uint64_t>(localCount_, chunk * capacity);
                        uint64_t count = std::min<uint64_t>(capacity, localCount_ - begin);
                        const T* data  = count ? source + firstIndex_ + begin : &emptyValue;
                        component.storeChunkRaw(data, {globalOffset_ + begin}, {count});
                        flushSeries();
                    }
                };
                if (key == "x" || key == "y" || key == "z")
                {
                    auto record = species["position"];
                    record.setUnitDimension({1, 0, 0, 0, 0, 0, 0});
                    if (!positionOffsetWritten_)
                    {
                        setPositionOffset<T>(iteration, globalCount_);
                        positionOffsetWritten_ = true;
                    }
                    store(record[key]);
                }
                else
                {
                    auto record = species[key];
                    record.setUnitDimension({0, 0, 0, 0, 0, 0, 0});
                    store(record[openPMD::RecordComponent::SCALAR]);
                }
            },
            field);
    }

    uint64_t localNumParticles() override { return localCount_; }
    uint64_t globalNumParticles() override { return globalCount_; }

    void closeStep() override
    {
        if (series_)
        {
            if (activeIteration_)
            {
                series_->iterations[iterationIndex_].close();
                activeIteration_ = false;
            }
            else { flushSeries(); }
            series_->close();
            series_.reset();
        }
        pathStep_.clear();
    }

private:
    void requireSeries(const std::string& key) const
    {
        if (!series_) { throw std::runtime_error("Cannot write " + key + ": OpenPMD series is not open"); }
    }

    void requireIteration(const std::string& key) const
    {
        requireSeries(key);
        if (!activeIteration_)
        {
            throw std::runtime_error("Cannot write " + key + ": OpenPMD checkpoint iteration is not active");
        }
    }

    void openSeries(const std::string& path)
    {
        requireNewPath(path);
        if (series_) { return; }

        int directoryError = 0;
        if (bp5_ && rank_ == 0)
        {
            std::error_code error;
            std::filesystem::create_directories(path, error);
            directoryError = error ? 1 : 0;
        }
        MPI_Bcast(&directoryError, 1, MPI_INT, 0, comm_);
        if (directoryError) { throw std::runtime_error("Cannot create OpenPMD output directory " + path); }

        const auto access = createdPaths_.count(path) ? openPMD::Access::APPEND : openPMD::Access::CREATE;
        const auto config = backendConfiguration(bp5_, collective_, config_, engineConfig_, false);
        series_           = std::make_unique<openPMD::Series>(seriesPath(path, bp5_), access, comm_, config);
        series_->setAttribute("sphexa_schema", uint32_t{1});
        for (const auto& [key, values] : pendingFileAttributes_)
        {
            std::visit([&](const auto& value) { writePendingAttribute(key, value); }, values);
        }
        for (const auto& [key, value] : pendingStringAttributes_)
        {
            series_->setAttribute(key, value);
        }
        pendingFileAttributes_.clear();
        pendingStringAttributes_.clear();
        createdPaths_.insert(path);
    }

    void requireNewPath(const std::string& path) const
    {
        if (series_ && path != pathStep_)
        {
            throw std::runtime_error("OpenPMD writer cannot switch paths before closeStep");
        }
    }

    void flushSeries()
    {
        if (series_) { series_->flush(bp5_ ? R"({"adios2":{"engine":{"preferred_flush_target":"disk"}}})" : "{}"); }
    }

    template<class T>
    void writePendingAttribute(const std::string& key, const std::vector<T>& value)
    {
        if (value.size() == 1) { series_->setAttribute(key, value.front()); }
        else { series_->setAttribute(key, value); }
    }

    MPI_Comm    comm_{MPI_COMM_NULL};
    int         rank_{0}, numRanks_{0};
    bool        bp5_{false};
    size_t      stagingBytes_;
    std::string collective_;
    std::string config_;
    std::string engineConfig_;

    std::unique_ptr<openPMD::Series>        series_;
    bool                                    activeIteration_{false};
    uint64_t                                iterationIndex_{0};
    bool                                    positionOffsetWritten_{false};
    std::string                             pathStep_;
    size_t                                  firstIndex_{0};
    uint64_t                                localCount_{0}, globalCount_{0}, globalOffset_{0};
    std::map<std::string, uint64_t>         nextIteration_;
    std::set<std::string>                   createdPaths_;
    std::map<std::string, PendingAttribute> pendingFileAttributes_;
    std::map<std::string, std::string>      pendingStringAttributes_;
};

class OpenPMDReader final : public IFileReader
{
public:
    using Base      = IFileReader;
    using FieldType = typename Base::FieldType;

    OpenPMDReader(MPI_Comm comm, bool bp5)
        : comm_(comm)
        , bp5_(bp5)
    {
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &numRanks_);
    }

    ~OpenPMDReader() override { closeStep(); }

    int     rank() const override { return rank_; }
    int64_t numParticles() const override { return globalCount_; }

    void setStep(std::string path, int step, FileMode mode) override
    {
        closeStep();
        const auto config = backendConfiguration(bp5_, "all", "{}", "{}", true);
        series_ = std::make_unique<openPMD::Series>(seriesPath(path, bp5_), openPMD::Access::READ_ONLY, comm_, config);
        if (!series_->containsAttribute("sphexa_schema") ||
            series_->getAttribute("sphexa_schema").template get<uint32_t>() != 1)
        {
            throw std::runtime_error("Unsupported SPH-EXA OpenPMD schema in " + path);
        }

        if (series_->iterations.empty())
        {
            globalCount_ = 0;
            localCount_  = 0;
            firstIndex_  = 0;
            lastIndex_   = 0;
            return;
        }

        auto iteration = series_->iterations.begin();
        if (step < 0) { iteration = std::prev(series_->iterations.end()); }
        else
        {
            if (size_t(step) >= series_->iterations.size()) { throw std::out_of_range("OpenPMD step does not exist"); }
            std::advance(iteration, step);
        }
        iterationIndex_    = iteration->first;
        auto openIteration = series_->iterations[iterationIndex_];
        openIteration.open();

        globalCount_ = particleCount(openIteration);
        if (mode == FileMode::collective)
        {
            std::tie(firstIndex_, lastIndex_) =
                std::pair<int64_t, int64_t>{globalCount_ * rank_ / numRanks_, globalCount_ * (rank_ + 1) / numRanks_};
        }
        else { firstIndex_ = 0, lastIndex_ = globalCount_; }
        localCount_ = lastIndex_ - firstIndex_;
    }

    std::vector<std::string> fileAttributes() override { return series_->attributes(); }
    std::vector<std::string> stepAttributes() override { return series_->iterations[iterationIndex_].attributes(); }

    int64_t fileAttributeSize(const std::string& key) override { return attributeSize(series_->getAttribute(key)); }

    int64_t stepAttributeSize(const std::string& key) override
    {
        return attributeSize(series_->iterations[iterationIndex_].getAttribute(key));
    }

    void fileAttribute(const std::string& key, FieldType values, int64_t size) override
    {
        std::visit([&](auto pointer) { readAttribute(*series_, key, pointer, size); }, values);
    }
    void fileAttribute(const std::string& key, std::string& value) override
    {
        value = series_->getAttribute(key).template get<std::string>();
    }

    void stepAttribute(const std::string& key, FieldType values, int64_t size) override
    {
        auto iteration = series_->iterations[iterationIndex_];
        std::visit([&](auto pointer) { readAttribute(iteration, key, pointer, size); }, values);
    }
    void stepAttribute(const std::string& key, std::string& value) override
    {
        value = series_->iterations[iterationIndex_].getAttribute(key).template get<std::string>();
    }

    void readField(const std::string& key, FieldType field) override
    {
        auto species = series_->iterations[iterationIndex_].particles[particleSpecies];
        auto load    = [&](auto component)
        {
            std::visit(
                [&](auto* destination)
                {
                    component.loadChunkRaw(destination, {uint64_t(firstIndex_)}, {uint64_t(localCount_)});
                    series_->flush();
                },
                field);
        };
        if (key == "x" || key == "y" || key == "z") { load(species["position"][key]); }
        else { load(species[key][openPMD::RecordComponent::SCALAR]); }
    }

    uint64_t localNumParticles() override { return uint64_t(localCount_); }
    uint64_t globalNumParticles() override { return uint64_t(globalCount_); }

    void closeStep() override
    {
        if (series_)
        {
            series_->close();
            series_.reset();
        }
    }

private:
    static int64_t particleCount(openPMD::Iteration& iteration)
    {
        auto species = iteration.particles[particleSpecies];
        if (species.contains("position"))
        {
            auto position = species["position"];
            for (const char* component : {"x", "y", "z"})
            {
                if (position.contains(component)) { return int64_t(position[component].getExtent()[0]); }
            }
        }
        for (auto& [name, record] : species)
        {
            (void)name;
            if (record.contains(openPMD::RecordComponent::SCALAR))
            {
                return int64_t(record[openPMD::RecordComponent::SCALAR].getExtent()[0]);
            }
        }
        return 0;
    }

    MPI_Comm comm_{MPI_COMM_NULL};
    int      rank_{0}, numRanks_{0};
    bool     bp5_{false};

    std::unique_ptr<openPMD::Series> series_;
    uint64_t                         iterationIndex_{0};
    int64_t                          firstIndex_{0}, lastIndex_{0}, localCount_{0}, globalCount_{0};
};

std::unique_ptr<IFileWriter> makeOpenPMDWriter(MPI_Comm comm, bool bp5, size_t stagingBytes,
                                               const std::string& collective, const std::string& config,
                                               const std::string& engineConfig)
{
    return std::make_unique<OpenPMDWriter>(comm, bp5, stagingBytes, collective, config, engineConfig);
}

std::unique_ptr<IFileReader> makeOpenPMDReader(MPI_Comm comm, bool bp5)
{
    return std::make_unique<OpenPMDReader>(comm, bp5);
}

} // namespace sphexa

#else

namespace sphexa
{

std::unique_ptr<IFileWriter> makeOpenPMDWriter(MPI_Comm, bool, size_t, const std::string&, const std::string&,
                                               const std::string&)
{
    throw std::runtime_error("OpenPMD I/O requested, but SPH-EXA was built without openPMD support");
}

std::unique_ptr<IFileReader> makeOpenPMDReader(MPI_Comm, bool)
{
    throw std::runtime_error("OpenPMD I/O requested, but SPH-EXA was built without openPMD support");
}

} // namespace sphexa

#endif
