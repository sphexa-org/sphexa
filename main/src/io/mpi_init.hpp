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
 * @brief MPI initialization matching the threading level required by the
 *        selected HDF5 VFD
 *
 * HDF5's subfiling VFD requires the application to initialize MPI with
 * MPI_Init_thread() at the MPI_THREAD_MULTIPLE level; with a plain MPI_Init(),
 * H5OpenFile() fails with
 * "The HDF5 subfiling VFD requires that the application initialized MPI with
 * MPI_Init_thread (MPI_THREAD_MULTIPLE)".
 *
 * H5hut selects the subfiling VFD from the environment variable
 * H5HUT_VFD=subfiling (see h5core/h5_file.c), i.e. the VFD choice is a runtime
 * property of the job, invisible to the code that calls MPI_Init(). The
 * initMpi() helper below reads the same environment variable and raises the
 * requested threading level accordingly, so the same binary works with and
 * without subfiling.
 *
 * @author SPH-EXA developers
 */

#include <cstdlib>
#include <cstring>
#include <iostream>

#include <mpi.h>

namespace sphexa
{

//! @brief true if H5hut's H5HUT_VFD environment switch selects the subfiling VFD
/*! Must match H5hut's own check in h5core/h5_file.c: an exact string
 * comparison against "subfiling". */
inline bool subfilingVfdRequested()
{
    const char* vfd = getenv("H5HUT_VFD");
    return vfd != nullptr && std::strcmp(vfd, "subfiling") == 0;
}

//! @brief MPI_Init that requests MPI_THREAD_MULTIPLE when the HDF5 subfiling
//! VFD is selected, MPI_THREAD_SINGLE otherwise
/*! Either argument may be NULL, exactly like MPI_Init(NULL, NULL). A warning
 * is printed if MPI cannot provide the requested level, since subfiling will
 * fail later at file open with a less obvious error message. */
inline void initMpi(int* argc, char*** argv)
{
    int requested = subfilingVfdRequested() ? MPI_THREAD_MULTIPLE : MPI_THREAD_SINGLE;
    int provided  = MPI_THREAD_SINGLE;
    MPI_Init_thread(argc, argv, requested, &provided);
    if (provided < requested)
    {
        std::cerr << "WARNING: MPI provides thread level " << provided << ", requested " << requested
                  << (requested == MPI_THREAD_MULTIPLE ? " (required by the HDF5 subfiling VFD)\n" : "\n");
    }
}

} // namespace sphexa
