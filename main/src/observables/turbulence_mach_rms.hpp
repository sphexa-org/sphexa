/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
 *               2021 University of Basel
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

#include <mpi.h>

#include "cstone/primitives/accel_switch.hpp"

#include "io/file_utils.hpp"
#include "conserved_quantities.hpp"
#include "gpu_reductions.h"
#include "iobservables.hpp"

namespace sphexa
{

template<class Dataset>
double localMachSquareSum(size_t first, size_t last, Dataset& d)
{
    const auto* vx = d.vx.data();
    const auto* vy = d.vy.data();
    const auto* vz = d.vz.data();
    const auto* c  = d.c.data();

    double localMachSquareSum = 0.0;

#pragma omp parallel for reduction(+ : localMachSquareSum)
    for (size_t i = first; i < last; ++i)
    {
        util::array<double, 3> V{vx[i], vy[i], vz[i]};
        localMachSquareSum += norm2(V) / (c[i] * c[i]);
    }

    return localMachSquareSum;
}

/*!@brief global calculation of the Mach number RMS
 *
 * @tparam Dataset
 * @param[in]     first     first locally assigned particle index of buffers in @p d
 * @param[in]     last      last locally assigned particle index of buffers in @p d
 * @param d                 particle dataset
 * @param comm              MPI communication rank
 * @return
 */
template<class Dataset>
double calculateMachRMS(size_t first, size_t last, Dataset& d, MPI_Comm comm)
{
    double localMachRms;

    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        localMachRms = machSquareSumGpu(rawPtr(d.devData.vx), rawPtr(d.devData.vy), rawPtr(d.devData.vz),
                                        rawPtr(d.devData.c), first, last);
    }
    else { localMachRms = localMachSquareSum(first, last, d); }

    int    rootRank = 0;
    double globalMachRms;

    MPI_Reduce(&localMachRms, &globalMachRms, 1, MPI_DOUBLE, MPI_SUM, rootRank, comm);

    return std::sqrt(globalMachRms / d.numParticlesGlobal);
}

//! @brief Observables that includes times, energies and the root mean square of the mach number
template<class Dataset>
class TurbulenceMachRMS : public IObservables<Dataset>
{
    std::ostream& constantsFile;

public:
    TurbulenceMachRMS(std::ostream& constPath)
        : constantsFile(constPath)
    {
    }

    using T = typename Dataset::RealType;

    void computeAndWrite(Dataset& simData, size_t firstIndex, size_t lastIndex, const cstone::Box<T>& /*box*/)
    {
        auto& d = simData.hydro;
        computeConservedQuantities(firstIndex, lastIndex, d, simData.comm);
        double machRms = calculateMachRMS(firstIndex, lastIndex, d, simData.comm);

        int rank;
        MPI_Comm_rank(simData.comm, &rank);

        if (rank == 0)
        {
            fileutils::writeColumns(constantsFile, ' ', d.iteration, d.ttot, d.minDt, d.etot, d.ecin, d.eint, d.egrav,
                                    d.linmom, d.angmom, machRms);
        }
    }
};

} // namespace sphexa