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
 * @brief Shared runtime options for SPH-EXA file output backends
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

#include "io/arg_parser.hpp"

namespace sphexa
{

enum class IOBackend
{
    h5hut,
    openpmdHdf5,
    openpmdBp5,
    ascii
};

struct IOOptions
{
    IOBackend   backend          = IOBackend::h5hut;
    std::string collective       = "all";
    uint64_t    stagingBytes     = 64 * 1024 * 1024;
    std::string openpmdConfig    = "{}";
    std::string openpmdEngineCfg = "{}";
};

inline IOBackend ioBackendFromString(const std::string& backend)
{
    if (backend == "h5hut") { return IOBackend::h5hut; }
    if (backend == "openpmd" || backend == "openpmd-hdf5") { return IOBackend::openpmdHdf5; }
    if (backend == "bp5" || backend == "openpmd-bp5") { return IOBackend::openpmdBp5; }
    if (backend == "ascii") { return IOBackend::ascii; }
    throw std::runtime_error("Unknown I/O backend '" + backend +
                             "': expected h5hut, openpmd-hdf5, openpmd-bp5 or ascii");
}

inline std::string ioBackendName(IOBackend backend)
{
    switch (backend)
    {
        case IOBackend::h5hut: return "h5hut";
        case IOBackend::openpmdHdf5: return "openpmd-hdf5";
        case IOBackend::openpmdBp5: return "openpmd-bp5";
        case IOBackend::ascii: return "ascii";
    }
    return "unknown";
}

inline std::string validateCollectiveMode(std::string mode)
{
    if (mode == "collective") { return "data"; }
    if (mode == "independent") { return "none"; }
    if (mode != "all" && mode != "data" && mode != "metadata" && mode != "none")
    {
        throw std::runtime_error("Unknown I/O collective mode '" + mode + "': expected all, data, metadata or none");
    }
    return mode;
}

inline IOOptions ioOptionsFromArguments(const ArgParser& parser)
{
    IOOptions options;

    const std::string backend =
        parser.get("--io-backend", parser.exists("--ascii") ? std::string("ascii") : std::string("h5hut"));
    options.backend = ioBackendFromString(backend);

    if (parser.exists("--ascii") && options.backend != IOBackend::ascii)
    {
        throw std::runtime_error("--ascii cannot be combined with --io-backend=" + backend);
    }

    std::string collective;
    if (parser.exists("--collective")) { collective = parser.get("--collective"); }
    else if (parser.exists("--hdf5-collective")) { collective = parser.get("--hdf5-collective"); }
    else if (const char* env = std::getenv("SPH_EXA_IO_COLLECTIVE"); env && *env) { collective = env; }
    else if (const char* env = std::getenv("SPH_EXA_HDF5_COLLECTIVE_MODE"); env && *env) { collective = env; }
    else { collective = "all"; }
    options.collective = validateCollectiveMode(std::move(collective));

    size_t           stagingMiB = parser.get<size_t>("--staging-mib", 64);
    constexpr size_t MiB        = 1024 * 1024;
    if (stagingMiB > std::numeric_limits<size_t>::max() / MiB)
    {
        throw std::runtime_error("--staging-mib overflows size_t");
    }
    options.stagingBytes = uint64_t(stagingMiB * MiB);

    options.openpmdConfig = parser.get("--openpmd-config", std::string("{}"));
    options.openpmdEngineCfg =
        parser.get("--openpmd-engine-config", std::getenv("SPH_EXA_OPENPMD_ENGINE_CONFIG")
                                                  ? std::string(std::getenv("SPH_EXA_OPENPMD_ENGINE_CONFIG"))
                                                  : std::string("{}"));

    return options;
}

inline void configureIOEnvironment(const IOOptions& options)
{
    setenv("SPH_EXA_HDF5_COLLECTIVE_MODE", options.collective.c_str(), 1);
    const bool collectiveMetadata = options.collective == "all" || options.collective == "metadata";
    setenv("OPENPMD_HDF5_COLLECTIVE_METADATA", collectiveMetadata ? "ON" : "OFF", 1);
}

} // namespace sphexa
