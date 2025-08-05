#pragma once

#include <chrono>
#include <tuple>
#include <omp.h>
#include "version.h"

namespace sphexa
{

auto initMpi()
{
    int rank     = 0;
    int numRanks = 0;

    auto t0      = std::chrono::system_clock::now();
    int  t_main  = std::chrono::duration_cast<std::chrono::seconds>(t0.time_since_epoch()).count();
    int  t_slurm = getenv("SLURM_JOB_START_TIME") == nullptr ? t_main : std::stoi(getenv("SLURM_JOB_START_TIME"));

    auto t1 = std::chrono::high_resolution_clock::now();
    MPI_Init(NULL, NULL);
    auto  t2          = std::chrono::high_resolution_clock::now();
    float mpiInitTime = std::chrono::duration<float>(t2 - t1).count();

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    if (rank == 0)
    {
        int mpi_version, mpi_subversion;
        printf("# SPHEXA: %s/%s\n", GIT_BRANCH, GIT_COMMIT_HASH);
        MPI_Get_version(&mpi_version, &mpi_subversion);
#ifdef _OPENMP
        printf("# %d MPI-%d.%d process(es) with %d OpenMP-%u thread(s)/process, SLURM init time %d s, "
               "MPI_Init time %f s\n",
               numRanks, mpi_version, mpi_subversion, omp_get_max_threads(), _OPENMP, t_main - t_slurm, mpiInitTime);
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
