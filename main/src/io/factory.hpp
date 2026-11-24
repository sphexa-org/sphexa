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
 * @brief file I/O interface
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <memory>

#include "ifile_io_impl.h"
#include "io_options.hpp"

namespace sphexa
{

std::unique_ptr<IFileWriter> fileWriterFactory(bool ascii, MPI_Comm comm, bool sequential = false)
{
    if (ascii) { return makeAsciiWriter(comm); }
    else
    {
        if (sequential) { return makeH5PartWriterSeq(comm); }
        else { return makeH5PartWriter(comm); }
    }
}

std::unique_ptr<IFileReader> fileReaderFactory(bool /*ascii*/, MPI_Comm comm) { return makeH5PartReader(comm); }

std::unique_ptr<IFileWriter> fileWriterFactory(const IOOptions& options, MPI_Comm comm)
{
    configureIOEnvironment(options);
    switch (options.backend)
    {
        case IOBackend::ascii: return makeAsciiWriter(comm);
        case IOBackend::h5hut: return makeH5PartWriter(comm);
        case IOBackend::openpmdHdf5:
            return makeOpenPMDWriter(comm, false, options.stagingBytes, options.collective, options.openpmdConfig,
                                     options.openpmdEngineCfg);
        case IOBackend::openpmdBp5:
            return makeOpenPMDWriter(comm, true, options.stagingBytes, options.collective, options.openpmdConfig,
                                     options.openpmdEngineCfg);
    }
    throw std::runtime_error("Unknown I/O backend");
}

std::unique_ptr<IFileReader> fileReaderFactory(const IOOptions& options, MPI_Comm comm)
{
    switch (options.backend)
    {
        case IOBackend::ascii: return makeH5PartReader(comm);
        case IOBackend::h5hut: return makeH5PartReader(comm);
        case IOBackend::openpmdHdf5: return makeOpenPMDReader(comm, false);
        case IOBackend::openpmdBp5: return makeOpenPMDReader(comm, true);
    }
    throw std::runtime_error("Unknown I/O backend");
}

} // namespace sphexa
