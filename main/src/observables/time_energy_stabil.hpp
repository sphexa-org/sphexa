/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
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

/*! @file
 * @brief output and calculate energies together with central density and stellar radius
 *
 * @author Lukas Schmidt
 * @author Ruben Cabezon
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mpi.h>
#include <vector>

#include "conserved_quantities.hpp"
#include "gpu_reductions.h"
#include "iobservables.hpp"
#include "io/file_utils.hpp"

namespace sphexa
{

inline constexpr size_t stabilSampleSize = 50;

struct greater
{
    template<class T>
    bool operator()(T const& a, T const& b) const
    {
        return a > b;
    }
};

template<class T>
struct StabilData
{
    std::array<T, stabilSampleSize> centralRadii;
    std::array<T, stabilSampleSize> centralDensities;
    std::array<T, stabilSampleSize> surfaceRadii;
};

/*! @brief local calculation of the central density candidates and stellar surface radius candidates
 *
 * @tparam        T            double or float
 * @param[in]     startIndex   first locally assigned particle index of buffers in @p d
 * @param[in]     endIndex     last locally assigned particle index of buffers in @p d
 * @param[in]     x            X coordinate array
 * @param[in]     y            Y coordinate array
 * @param[in]     z            Z coordinate array
 * @param[in]     rho          baryonic density
 *
 * Returns the densities of the 50 local particles with the smallest radii and the
 * radii of the 50 local particles with the largest radii.
 */
template<class T>
StabilData<T> localStabil(size_t startIndex, size_t endIndex, const T* x, const T* y, const T* z, const T* kx,
                          const T* m, const T* xm)
{
    using RadiusDensity = std::pair<T, T>;

    size_t localCount = endIndex - startIndex;

    std::vector<RadiusDensity> centralCandidates(localCount);
    std::vector<T>             surfaceCandidates(localCount);

    StabilData<T> result;
    result.centralRadii.fill(std::numeric_limits<T>::max());
    result.centralDensities.fill(T(0));
    result.surfaceRadii.fill(std::numeric_limits<T>::lowest());

#pragma omp parallel for
    for (size_t offset = 0; offset < localCount; offset++)
    {
        size_t i     = startIndex + offset;
        T      radius = std::sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
        T      density = kx[i] * m[i] / xm[i];

        centralCandidates[offset] = {radius, density};
        surfaceCandidates[offset] = radius;
    }

    std::sort(centralCandidates.begin(), centralCandidates.end(),
              [](const RadiusDensity& a, const RadiusDensity& b) { return a.first < b.first; });
    std::sort(surfaceCandidates.begin(), surfaceCandidates.end(), greater());

    size_t numCenterCandidates  = std::min(localCount, stabilSampleSize);
    size_t numSurfaceCandidates = std::min(localCount, stabilSampleSize);

    for (size_t i = 0; i < numCenterCandidates; ++i)
    {
        result.centralRadii[i]     = centralCandidates[i].first;
        result.centralDensities[i] = centralCandidates[i].second;
    }

    for (size_t i = 0; i < numSurfaceCandidates; ++i)
    {
        result.surfaceRadii[i] = surfaceCandidates[i];
    }

    return result;
}

/*! @brief global calculation of the central density and radius
 *
 * @tparam        T            double or float
 * @tparam        Dataset
 * @param[in]     startIndex   first locally assigned particle index of buffers in @p d
 * @param[in]     endIndex     last locally assigned particle index of buffers in @p d
 * @param[in]     d            particle data set
 */
template<typename T, class Dataset>
util::tuple<T, T> computeStabil(size_t startIndex, size_t endIndex, Dataset& d, MPI_Comm comm)
{
    if (d.kx.empty())
    {
        throw std::runtime_error("kx was empty. TimeEnergyStabil is only supported with volume elements (--prop ve)\n");
    }

    StabilData<T> localStabilData;

    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        auto [centralRadii, centralDensities, surfaceRadii] =
            gpuStabilLocal(rawPtr(d.devData.x), rawPtr(d.devData.y), rawPtr(d.devData.z), rawPtr(d.devData.kx),
                           rawPtr(d.devData.m), rawPtr(d.devData.xm), startIndex, endIndex, stabilSampleSize);

        localStabilData.centralRadii.fill(std::numeric_limits<T>::max());
        localStabilData.centralDensities.fill(T(0));
        localStabilData.surfaceRadii.fill(std::numeric_limits<T>::lowest());

        size_t numCenterCandidates  = std::min(centralRadii.size(), size_t(stabilSampleSize));
        size_t numSurfaceCandidates = std::min(surfaceRadii.size(), size_t(stabilSampleSize));

        for (size_t i = 0; i < numCenterCandidates; ++i)
        {
            localStabilData.centralRadii[i]     = T(centralRadii[i]);
            localStabilData.centralDensities[i] = T(centralDensities[i]);
        }

        for (size_t i = 0; i < numSurfaceCandidates; ++i)
        {
            localStabilData.surfaceRadii[i] = T(surfaceRadii[i]);
        }
    }
    else
    {
        localStabilData = localStabil(startIndex, endIndex, d.x.data(), d.y.data(), d.z.data(), d.kx.data(),
                                      d.m.data(), d.xm.data());
    }

    int rootRank = 0;
    int mpiranks = 1;
    MPI_Comm_size(comm, &mpiranks);

    int rank;
    MPI_Comm_rank(comm, &rank);

    size_t rootsize = stabilSampleSize * mpiranks;

    std::vector<T> globalCentralRadii(rank == rootRank ? rootsize : 0);
    std::vector<T> globalCentralDensities(rank == rootRank ? rootsize : 0);
    std::vector<T> globalSurfaceRadii(rank == rootRank ? rootsize : 0);

    MPI_Gather(localStabilData.centralRadii.data(), stabilSampleSize, MpiType<T>{},
               rank == rootRank ? globalCentralRadii.data() : nullptr, stabilSampleSize, MpiType<T>{}, rootRank,
               comm);
    MPI_Gather(localStabilData.centralDensities.data(), stabilSampleSize, MpiType<T>{},
               rank == rootRank ? globalCentralDensities.data() : nullptr, stabilSampleSize, MpiType<T>{}, rootRank,
               comm);
    MPI_Gather(localStabilData.surfaceRadii.data(), stabilSampleSize, MpiType<T>{},
               rank == rootRank ? globalSurfaceRadii.data() : nullptr, stabilSampleSize, MpiType<T>{}, rootRank,
               comm);

    T centralDensity = 0.;
    T radius         = 0.;

    if (rank == 0)
    {
        using RadiusDensity = std::pair<T, T>;
        std::vector<RadiusDensity> centerCandidates;
        std::vector<T>             surfaceCandidates;
        centerCandidates.reserve(rootsize);
        surfaceCandidates.reserve(rootsize);

        for (size_t i = 0; i < rootsize; ++i)
        {
            if (globalCentralRadii[i] != std::numeric_limits<T>::max())
            {
                centerCandidates.push_back({globalCentralRadii[i], globalCentralDensities[i]});
            }

            if (globalSurfaceRadii[i] != std::numeric_limits<T>::lowest()) { surfaceCandidates.push_back(globalSurfaceRadii[i]); }
        }

        std::sort(centerCandidates.begin(), centerCandidates.end(),
                  [](const RadiusDensity& a, const RadiusDensity& b) { return a.first < b.first; });
        std::sort(surfaceCandidates.begin(), surfaceCandidates.end(), greater());

        size_t numCenterCandidates  = std::min(centerCandidates.size(), stabilSampleSize);
        size_t numSurfaceCandidates = std::min(surfaceCandidates.size(), stabilSampleSize);

        for (size_t i = 0; i < numCenterCandidates; ++i) { centralDensity += centerCandidates[i].second; }
        for (size_t i = 0; i < numSurfaceCandidates; ++i) { radius += surfaceCandidates[i]; }

        if (numCenterCandidates > 0) { centralDensity /= T(numCenterCandidates); }
        if (numSurfaceCandidates > 0) { radius /= T(numSurfaceCandidates); }
    }

    return {centralDensity, radius};
}

//! @brief Observables that include times, energies, central density and stellar radius
template<class Dataset>
class TimeEnergyStabil : public IObservables<Dataset>
{
    std::ostream& constantsFile;

public:
    TimeEnergyStabil(std::ostream& constPath)
        : constantsFile(constPath)
    {
    }

    using T = typename Dataset::RealType;

    void computeAndWrite(Dataset& simData, size_t firstIndex, size_t lastIndex, const cstone::Box<T>& box)
    {
        auto& d = simData.hydro;
        computeConservedQuantities(firstIndex, lastIndex, d, simData.comm);
        auto [centralDensity, radius] = computeStabil<T>(firstIndex, lastIndex, d, simData.comm);
        int rank;
        MPI_Comm_rank(simData.comm, &rank);

        std::cout << "time_energy_stabil" << std::endl;

        if (rank == 0)
        {
            fileutils::writeColumns(
                constantsFile, ' ', d.iteration, d.ttot, d.minDt, d.etot, d.ecin, d.eint, d.egrav, d.linmom, d.angmom, centralDensity, radius);
        }
    }
};

} // namespace sphexa
