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
 * @brief Standalone file-IO benchmark
 *
 * Uses the same particle data, domain synchronization, checkpoint, and output
 * field machinery as the main SPH-EXA application. Physics is replaced by a
 * cheap acceleration-free particle drift between checkpoints.
 *
 * @author SPH-EXA developers
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include <mpi.h>

#include "cstone/domain/domain.hpp"

#include "init/settings.hpp"
#include "io/arg_parser.hpp"
#include "io/checkpoint.hpp"
#include "io/factory.hpp"
#include "io/mpi_init.hpp"
#include "propagator/factory.hpp"
#include "sphexa/simulation_data.hpp"

namespace sphexa
{

//! @brief deterministic pseudo-random double in [0, 1) from an integer
double random01(uint64_t i)
{
    uint64_t h = i * 0x9E3779B97F4A7C15u;
    h          = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9u;
    h          = (h ^ (h >> 27)) * 0x94D049BB133111EBu;
    h          = h ^ (h >> 31);
    return double(h >> 11) / 9007199254740992.0;
}

template<class Dataset>
void initFields(Dataset& d, size_t localCount, uint64_t globalOffset, double boxSize, double dt)
{
    d.resize(localCount);

    auto fields = d.data();
    for (size_t fieldIndex = 0; fieldIndex < fields.size(); ++fieldIndex)
    {
        if (!d.isAllocated(fieldIndex)) { continue; }

        uint64_t seed = 0;
        for (char c : std::string(d.fieldNames[fieldIndex]))
        {
            seed = seed * 131 + uint8_t(c);
        }

        std::visit(
            [localCount, globalOffset, seed](auto* field)
            {
                using T = typename std::decay_t<decltype(*field)>::value_type;
                if constexpr (std::is_floating_point_v<T>)
                {
                    for (size_t i = 0; i < localCount; ++i)
                    {
                        (*field)[i] = T(0.5 + 0.5 * random01(globalOffset + i + seed));
                    }
                }
                else { std::fill(field->begin(), field->end(), T{}); }
            },
            fields[fieldIndex]);
    }

    double particleSpacing = boxSize / std::cbrt(double(d.numParticlesGlobal));
    for (size_t i = 0; i < localCount; ++i)
    {
        uint64_t globalIndex = globalOffset + i;
        double   rx          = random01(globalIndex);
        double   ry          = random01(globalIndex + 0x100000000u);
        double   rz          = random01(globalIndex + 0x200000000u);

        d.x[i]  = boxSize * rx;
        d.y[i]  = boxSize * ry;
        d.z[i]  = boxSize * rz;
        d.vx[i] = float(0.02 * (rx - 0.5));
        d.vy[i] = float(0.02 * (ry - 0.5));
        d.vz[i] = float(0.02 * (rz - 0.5));

        d.x_m1[i]  = float(d.vx[i] * dt);
        d.y_m1[i]  = float(d.vy[i] * dt);
        d.z_m1[i]  = float(d.vz[i] * dt);
        d.du_m1[i] = 0;

        d.h[i]   = float(2.0 * particleSpacing);
        d.m[i]   = float(1.0 / double(d.numParticlesGlobal));
        d.u[i]   = 1.0 + rx;
        d.rho[i] = float(1.0 + rz);
        d.p[i]   = float(0.4 * d.u[i] * d.rho[i]);
        d.c[i]   = float(1.0 + rx);
        d.id[i]  = globalIndex;
    }
}

template<class Dataset, class T>
void driftParticles(Dataset& d, size_t first, size_t last, double dt, const cstone::Box<T>& box)
{
#pragma omp parallel for schedule(static)
    for (size_t i = first; i < last; ++i)
    {
        double dx = d.vx[i] * dt;
        double dy = d.vy[i] * dt;
        double dz = d.vz[i] * dt;

        auto position = cstone::putInBox(cstone::Vec3<T>{d.x[i] + dx, d.y[i] + dy, d.z[i] + dz}, box);

        d.x[i] = position[0];
        d.y[i] = position[1];
        d.z[i] = position[2];

        d.x_m1[i] = float(dx);
        d.y_m1[i] = float(dy);
        d.z_m1[i] = float(dz);
    }
}

template<class Dataset>
uint64_t checkpointBytes(Dataset& d)
{
    auto fields = d.data();

    uint64_t bytesPerParticle = 0;
    for (int fieldIndex : d.outputFieldIndices)
    {
        if (d.isAllocated(fieldIndex))
        {
            bytesPerParticle += std::visit(
                [](auto* field)
                {
                    using Vector = std::remove_pointer_t<decltype(field)>;
                    return uint64_t(sizeof(typename Vector::value_type));
                },
                fields[fieldIndex]);
        }
    }
    return bytesPerParticle * d.numParticlesGlobal;
}

std::string paddedIndex(size_t idx, size_t width)
{
    std::string s = std::to_string(idx);
    return std::string(width - std::min(width, s.size()), '0') + s;
}

} // namespace sphexa

int main(int argc, char** argv)
{
    sphexa::initMpi(&argc, &argv);

    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    using namespace sphexa;
    using Exec    = cstone::execution::Cpu;
    using Dataset = SimulationData<Exec>;
    using Domain  = cstone::Domain<sph::SphTypes::KeyType, sph::SphTypes::CoordinateType, Exec>;

    ArgParser parser(argc, const_cast<const char**>(argv));

    uint64_t numParticlesGlobal = 1000000;
    if (parser.exists("-n")) { numParticlesGlobal = std::stoull(parser.get("-n")); }
    size_t      numSteps     = parser.get<size_t>("-s", 10);
    size_t      writeFreq    = parser.get<size_t>("-w", 1);
    size_t      stepsPerFile = parser.get<size_t>("--steps-per-file", 1);
    std::string outFile      = parser.get("-o", std::string("iobench_dump"));
    bool        ascii        = parser.exists("--ascii");
    bool        overwrite    = parser.exists("--overwrite");

    std::string hdf5Mode = parser.get("--hdf5-collective", std::string(""));
    if (!hdf5Mode.empty())
    {
        if (hdf5Mode != "all" && hdf5Mode != "data" && hdf5Mode != "metadata" && hdf5Mode != "none" &&
            hdf5Mode != "collective" && hdf5Mode != "independent")
        {
            if (rank == 0)
            {
                std::cerr << "ERROR: invalid --hdf5-collective '" << hdf5Mode
                          << "', valid values are 'all', 'data', 'metadata' and 'none'\n";
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        setenv("SPH_EXA_HDF5_COLLECTIVE_MODE", hdf5Mode.c_str(), 1);
    }

    if (writeFreq == 0) { writeFreq = 1; }
    if (numSteps == 0) { numSteps = 1; }
    if (numParticlesGlobal == 0)
    {
        if (rank == 0) { std::cerr << "ERROR: the global particle count must be greater than zero\n"; }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::ofstream nullOutput("/dev/null");
    Dataset       simData;
    simData.comm = MPI_COMM_WORLD;

    auto propagator = propagatorFactory<Domain, Dataset>("std", false, nullOutput, rank, InitSettings{});
    propagator->activateFields(simData);

    std::vector<std::string> outputFields = {"x",  "y",  "z",    "h",    "m",    "u",     "vx",
                                             "vy", "vz", "x_m1", "y_m1", "z_m1", "du_m1", "id"};
    if (parser.exists("-f")) { outputFields = parser.getCommaList("-f"); }
    else { outputFields = propagator->conservedFields(); }
    simData.setOutputFields(outputFields);

    auto fileWriter = fileWriterFactory(ascii, MPI_COMM_WORLD);

    int dirStatus = 0;
    if (rank == 0)
    {
        std::filesystem::path outDir = std::filesystem::path(outFile).parent_path();
        if (!outDir.empty())
        {
            std::error_code ec;
            if (!std::filesystem::create_directories(outDir, ec) && ec)
            {
                std::cerr << "ERROR: cannot create output directory " << outDir << ": " << ec.message() << "\n";
                dirStatus = 1;
            }
        }
    }
    MPI_Bcast(&dirStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (dirStatus != 0) { MPI_Abort(MPI_COMM_WORLD, 1); }

    uint64_t firstParticle = numParticlesGlobal * uint64_t(rank) / uint64_t(numRanks);
    uint64_t lastParticle  = numParticlesGlobal * uint64_t(rank + 1) / uint64_t(numRanks);
    size_t   localCount    = size_t(lastParticle - firstParticle);

    double boxSize       = 1.0;
    double dt            = 0.01;
    auto&  d             = simData.hydro;
    d.numParticlesGlobal = numParticlesGlobal;
    d.minDt              = dt;
    d.minDt_m1           = dt;
    initFields(d, localCount, firstParticle, boxSize, dt);

    using T       = sph::SphTypes::CoordinateType;
    using BoxType = cstone::Box<T>;
    BoxType box(0, boxSize, cstone::BoundaryType::periodic);

    uint64_t bucketSizeFocus = 64;
    uint64_t bucketSize      = std::max(bucketSizeFocus, numParticlesGlobal / (100 * uint64_t(numRanks)));
    Domain   domain(cstone::execution::cpu, rank, numRanks, bucketSize, bucketSizeFocus, 1.0, MPI_COMM_WORLD, box);
    domain.setGrowthAllocRate(d.getAllocGrowthRate());

    MPI_Barrier(MPI_COMM_WORLD);
    double initStart = MPI_Wtime();
    propagator->sync(domain, simData);
    MPI_Barrier(MPI_COMM_WORLD);
    double initElapsed = MPI_Wtime() - initStart;

    uint64_t totalBytesPerDump = checkpointBytes(d);
    size_t   numFields         = d.outputFieldNames.size();

    if (rank == 0)
    {
        std::cout << "=== IO benchmark ===\n"
                  << "ranks: " << numRanks << ", particles (global): " << numParticlesGlobal << ", steps: " << numSteps
                  << ", dump every " << writeFreq << " step(s)\n"
                  << "steps per file: "
                  << (stepsPerFile == 0 ? std::string("single file") : std::to_string(stepsPerFile))
                  << ", writer: " << (ascii ? "ascii" : "h5 (parallel)") << (overwrite ? ", overwrite enabled" : "")
                  << ", hdf5-collective: "
                  << (std::getenv("SPH_EXA_HDF5_COLLECTIVE_MODE") ? std::getenv("SPH_EXA_HDF5_COLLECTIVE_MODE")
                                                                  : "all (default)")
                  << "\nfields (" << numFields << "): ";
        for (const auto& name : d.outputFieldNames)
        {
            std::cout << name << " ";
        }
        std::cout << "\nbytes per checkpoint: " << double(totalBytesPerDump) / (1024. * 1024.) << " MiB"
                  << "\ninitial domain sync: " << initElapsed << " s\n"
                  << std::endl;
    }

    InitSettings settings = {
        {"numParticlesGlobal", double(numParticlesGlobal)}, {"gravConstant", 0.0}, {"boxSize", boxSize}, {"dt", dt}};

    struct DumpStats
    {
        size_t      step;
        double      bytes;
        double      tMax, tAvg;
        std::string file;
    };
    std::vector<DumpStats> stats;

    std::ofstream csvStream;
    if (rank == 0)
    {
        csvStream.open("io_summary.csv");
        csvStream << "step,payload_bytes,io_write_max_s,io_write_mean_s,payload_GiB_per_s\n";
    }

    size_t numDumps    = 0;
    size_t lastFileIdx = size_t(-1);
    for (size_t step = 1; step <= numSteps; ++step)
    {
        driftParticles(d, domain.startIndex(), domain.endIndex(), dt, box);
        propagator->sync(domain, simData);

        d.iteration = step;
        d.ttot      = step * dt;

        if (step % writeFreq != 0) { continue; }

        ++numDumps;
        size_t      fileIdx = (stepsPerFile == 0) ? 0 : (numDumps - 1) / stepsPerFile;
        std::string path    = (stepsPerFile == 0) ? outFile + fileWriter->suffix()
                                                  : outFile + "_" + paddedIndex(fileIdx, 4) + fileWriter->suffix();

        if (fileIdx != lastFileIdx)
        {
            int fileStatus = 0;
            if (rank == 0)
            {
                if (std::filesystem::exists(path))
                {
                    if (overwrite)
                    {
                        std::filesystem::remove(path);
                        fileStatus = 1;
                    }
                    else
                    {
                        std::cerr << "ERROR: output file " << path
                                  << " already exists, would append steps. Delete it or pass --overwrite\n";
                        fileStatus = 2;
                    }
                }
            }
            MPI_Bcast(&fileStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (fileStatus == 2) { MPI_Abort(MPI_COMM_WORLD, 1); }

            writeSettings(settings, path, fileWriter.get());
            lastFileIdx = fileIdx;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        writeCheckpoint(fileWriter.get(), path, domain, simData, propagator.get(), box);

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        double localDt = t1 - t0, maxDt = 0, sumDt = 0;
        MPI_Reduce(&localDt, &maxDt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&localDt, &sumDt, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            double tAvg = sumDt / numRanks;
            double gbs  = double(totalBytesPerDump) / (1024. * 1024. * 1024.) / maxDt;
            std::cout << "dump " << numDumps << " (step " << step
                      << "): " << double(totalBytesPerDump) / (1024. * 1024.) << " MiB in " << maxDt
                      << " s (slowest rank), " << gbs << " GB/s\n";
            stats.push_back({step, double(totalBytesPerDump), maxDt, tAvg, path});
            csvStream << step << ',' << totalBytesPerDump << ',' << std::setprecision(10) << maxDt << ',' << tAvg << ','
                      << gbs << '\n';
            csvStream.flush();
        }
    }

    if (rank == 0 && !stats.empty())
    {
        double totalBytes = 0, totalTimeMax = 0, totalTimeAvg = 0;
        for (const auto& s : stats)
        {
            totalBytes += s.bytes;
            totalTimeMax += s.tMax;
            totalTimeAvg += s.tAvg;
        }
        std::cout << "\n=== IO benchmark summary ===\n"
                  << "dumps: " << stats.size() << ", total data: " << totalBytes / (1024. * 1024. * 1024.) << " GiB\n"
                  << "aggregate IO rate (sum of dump wall times, slowest rank per dump): "
                  << totalBytes / (1024. * 1024. * 1024.) / totalTimeMax << " GB/s\n"
                  << "aggregate IO rate (average rank): " << totalBytes / (1024. * 1024. * 1024.) / totalTimeAvg
                  << " GB/s" << std::endl;
    }

    fileWriter.reset();
    propagator.reset();
    MPI_Finalize();
    return 0;
}
