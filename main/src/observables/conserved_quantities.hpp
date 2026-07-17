/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
 *               2022 University of Basel
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
 * @brief Global and local energy, and linear and angular momentum reductions
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Aurelien Cavelan
 * @author Ruben Cabezon
 */

#pragma once

#include "mpi.h"

#include "cstone/util/array.hpp"
#include "cstone/primitives/primitives_gpu.h"
#include "sph/eos.hpp"
#include "conserved_gpu.h"
#include "gpu_reductions.h"

namespace sphexa
{

template<class Dataset>
auto localConservedQuantities(size_t startIndex, size_t endIndex, Dataset& d)
{
    const auto* x    = d.x.data();
    const auto* y    = d.y.data();
    const auto* z    = d.z.data();
    const auto* vx   = d.vx.data();
    const auto* vy   = d.vy.data();
    const auto* vz   = d.vz.data();
    const auto* m    = d.m.data();
    const auto* temp = d.temp.data();
    const auto* u    = d.u.data();

    util::array<double, 3> linmom{0.0, 0.0, 0.0};
    util::array<double, 3> angmom{0.0, 0.0, 0.0};

    double sharedCv = sph::idealGasCv(d.muiConst, d.gamma);
    bool   haveMui  = !d.mui.empty();

#pragma omp declare reduction(+ : util::array<double, 3> : omp_out += omp_in) initializer(omp_priv(omp_orig))

    double eKin = 0.0;
#pragma omp parallel for reduction(+ : eKin, linmom, angmom)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        util::array<double, 3> X{x[i], y[i], z[i]};
        util::array<double, 3> V{vx[i], vy[i], vz[i]};
        auto                   mi = m[i];

        eKin += mi * norm2(V);
        linmom += mi * V;
        angmom += mi * cross(X, V);
    }

    double eInt = 0.0;

    if (!d.u.empty())
    {
#pragma omp parallel for reduction(+ : eInt)
        for (size_t i = startIndex; i < endIndex; i++)
        {
            auto mi = m[i];
            eInt += u[i] * mi;
        }
    }
    else if (!d.temp.empty())
    {

#pragma omp parallel for reduction(+ : eInt)
        for (size_t i = startIndex; i < endIndex; i++)
        {
            auto cv = haveMui ? sph::idealGasCv(d.mui[i], d.gamma) : sharedCv;
            auto mi = m[i];
            eInt += cv * temp[i] * mi;
        }
    }

    return std::make_tuple(0.5 * eKin, eInt, linmom, angmom);
}

inline std::size_t countErrBitsCpu(std::span<const std::uint64_t> id, std::size_t firstIndex, std::size_t lastIndex,
                                   unsigned errBit)
{
    std::uint64_t errMask = std::uint64_t{1} << errBit;

    std::size_t count{0};
#pragma omp parallel for schedule(static) reduction(+ : count)
    for (std::size_t i = firstIndex; i < lastIndex; i++)
    {
        if (bool(id[i] & errMask)) { count++; }
    }
    return count;
}

/*! @brief Computation of globally conserved quantities
 *
 * @tparam        T            float or double
 * @tparam        Dataset
 * @param[in]     startIndex   first locally assigned particle index of buffers in @p d
 * @param[in]     endIndex     last locally assigned particle index of buffers in @p d
 * @param[inout]  d            particle data set
 */
template<class Dataset>
void computeConservedQuantities(size_t startIndex, size_t endIndex, Dataset& d, MPI_Comm comm)
{
    double               eKin, eInt;
    cstone::Vec3<double> linmom, angmom;
    size_t               ncsum = 0, iadNumErrBits = 0;

    if constexpr (d.useGpu)
    {
        if (!d.nc.empty())
        {
            ncsum = cstone::reduce(cstone::execution::gpuDefaultStream, rawPtr(d.nc) + startIndex,
                                   endIndex - startIndex, size_t(0));
        }
        std::tie(eKin, eInt, linmom, angmom) = conservedQuantitiesGpu(
            sph::idealGasCv(d.muiConst, d.gamma), rawPtr(d.x), rawPtr(d.y), rawPtr(d.z), rawPtr(d.vx), rawPtr(d.vy),
            rawPtr(d.vz), rawPtr(d.temp), rawPtr(d.u), rawPtr(d.m), startIndex, endIndex);
    }
    else
    {
        if (!d.nc.empty())
        {
#pragma omp parallel for reduction(+ : ncsum)
            for (size_t i = startIndex; i < endIndex; i++)
            {
                ncsum += d.nc[i];
            }
        }

        std::tie(eKin, eInt, linmom, angmom) = localConservedQuantities(startIndex, endIndex, d);
    }
    d.localNeighbors = ncsum;

    if (d.iadConditionQuality > 0.0)
    {
        if (d.useGpu) { iadNumErrBits = countErrBitsGpu(d.id, startIndex, endIndex, d.iadRegBit); }
        else { iadNumErrBits = countErrBitsCpu(d.id, startIndex, endIndex, d.iadRegBit); }
    }

    util::array<double, 12> quantities, globalQuantities;
    std::fill(globalQuantities.begin(), globalQuantities.end(), double(0));

    quantities[0]  = eKin;
    quantities[1]  = eInt;
    quantities[2]  = d.egrav;
    quantities[3]  = linmom[0];
    quantities[4]  = linmom[1];
    quantities[5]  = linmom[2];
    quantities[6]  = angmom[0];
    quantities[7]  = angmom[1];
    quantities[8]  = angmom[2];
    quantities[9]  = double(ncsum);
    quantities[10] = double(endIndex - startIndex);
    quantities[11] = double(iadNumErrBits);

    MPI_Reduce(quantities.data(), globalQuantities.data(), quantities.size(), MpiType<double>{}, MPI_SUM, 0, comm);

    d.ecin  = globalQuantities[0];
    d.eint  = globalQuantities[1];
    d.egrav = globalQuantities[2];
    d.etot  = d.ecin + d.eint + d.egrav;

    util::array<double, 3> globalLinmom{globalQuantities[3], globalQuantities[4], globalQuantities[5]};
    util::array<double, 3> globalAngmom{globalQuantities[6], globalQuantities[7], globalQuantities[8]};
    d.linmom                 = std::sqrt(norm2(globalLinmom));
    d.angmom                 = std::sqrt(norm2(globalAngmom));
    d.totalNeighbors         = size_t(globalQuantities[9]);
    d.numParticlesGlobalPrev = d.numParticlesGlobal;
    d.numParticlesGlobal     = size_t(globalQuantities[10]);
    d.numIadRegBits          = size_t(globalQuantities[11]);
}

} // namespace sphexa
