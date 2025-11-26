#pragma once

#include <chrono>
#include <tuple>
#include <mpi.h>
#include <omp.h>
#include "version.h"

namespace sphexa
{

auto initMpi()
{
    int rank     = 0;
    int numRanks = 0;

    auto t0       = std::chrono::system_clock::now();
    int  ts_main  = std::chrono::duration_cast<std::chrono::seconds>(t0.time_since_epoch()).count();
    int  ts_slurm = getenv("SLURM_JOB_START_TIME") == nullptr ? ts_main : std::stoi(getenv("SLURM_JOB_START_TIME"));

    MPI_Init(NULL, NULL);
    auto t1      = std::chrono::high_resolution_clock::now();
    int  ts_init = std::chrono::duration_cast<std::chrono::seconds>(t1.time_since_epoch()).count();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int ts_local[4] = {ts_main, -ts_main, ts_init, -ts_init};
    int ts_reduce[4];
    MPI_Reduce(ts_local, ts_reduce, 4, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    int ts_main_min      = ts_reduce[0];
    int ts_main_max      = -ts_reduce[1];
    int delta_t_main_min = ts_main_min - ts_slurm;
    int delta_t_main_max = ts_main_max - ts_slurm;

    int delta_t_init_min = ts_reduce[2] - ts_main_max;
    int delta_t_init_max = -ts_reduce[3] - ts_main_max;

    if (rank == 0)
    {
        int mpi_version, mpi_subversion;
        printf("# SPHEXA: %s/%s\n", GIT_BRANCH, GIT_COMMIT_HASH);
        MPI_Get_version(&mpi_version, &mpi_subversion);
#ifdef _OPENMP
        printf("# %d MPI-%d.%d process(es) with %d OpenMP-%u thread(s)/process, SLURM init time %d-%d s, "
               "MPI_Init time %d-%d s\n",
               numRanks, mpi_version, mpi_subversion, omp_get_max_threads(), _OPENMP, delta_t_main_min,
               delta_t_main_max, delta_t_init_min, delta_t_init_max);
#else
        printf("# %d MPI-%d.%d process(es) without OpenMP\n", numRanks, mpi_version, mpi_subversion);
#endif
    }
    return std::make_tuple(rank, numRanks);
}

int exitSuccess()
{
    MPI_Finalize();
    return EXIT_SUCCESS;
}

} // namespace sphexa
