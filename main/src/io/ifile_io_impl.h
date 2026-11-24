#pragma once

#include <memory>
#include <mpi.h>

#include "ifile_io.hpp"

namespace sphexa
{

std::unique_ptr<IFileWriter> makeAsciiWriter(MPI_Comm comm);
std::unique_ptr<IFileWriter> makeH5PartWriter(MPI_Comm comm);
std::unique_ptr<IFileWriter> makeH5PartWriterSeq(MPI_Comm comm);
std::unique_ptr<IFileWriter> makeOpenPMDWriter(MPI_Comm comm, bool bp5, size_t stagingBytes,
                                               const std::string& collective, const std::string& config,
                                               const std::string& engineConfig);

std::unique_ptr<IFileReader> makeH5PartReader(MPI_Comm comm);
std::unique_ptr<IFileReader> makeOpenPMDReader(MPI_Comm comm, bool bp5);

} // namespace sphexa
