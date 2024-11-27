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
 * @brief Min-reduction to determine global timestep
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include <mpi.h>

#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/primitives/primitives_gpu.h"
#include "kernels.hpp"

namespace sph
{

//! @brief limit time-step based on accelerations when gravity is enabled
template<class Dataset>
auto accelerationTimestep(size_t first, size_t last, const Dataset& d)
{
    using T = typename Dataset::RealType;

    T maxAccSq = 0.0;
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        maxAccSq = cstone::maxNormSquareGpu(rawPtr(d.devData.ax) + first, rawPtr(d.devData.ay) + first,
                                            rawPtr(d.devData.az) + first, last - first);
    }
    else
    {
#pragma omp parallel for reduction(max : maxAccSq)
        for (size_t i = first; i < last; ++i)
        {
            cstone::Vec3<T> X{d.ax[i], d.ay[i], d.az[i]};
            maxAccSq = std::max(norm2(X), maxAccSq);
        }
    }

    return d.etaAcc * std::sqrt(d.eps / std::sqrt(maxAccSq));
}

//! @brief limit time-step based on divergence of velocity, this is called in the propagator when Divv is available
template<class Dataset>
auto rhoTimestep(size_t first, size_t last, const Dataset& d)
{
    using T = std::decay_t<decltype(d.divv[0])>;

    T maxDivv = -INFINITY;
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        if (d.devData.divv.empty()) { throw std::runtime_error("Divv needs to be available in rhoTimestep\n"); }
        auto minmax = cstone::MinMaxGpu<T>{}(rawPtr(d.devData.divv) + first, rawPtr(d.devData.divv) + last);
        maxDivv     = std::get<1>(minmax);
    }
    else
    {
        if (d.divv.empty()) { throw std::runtime_error("Divv needs to be available in rhoTimestep\n"); }

#pragma omp parallel for reduction(max : maxDivv)
        for (size_t i = first; i < last; ++i)
        {
            maxDivv = std::max(d.divv[i], maxDivv);
        }
    }
    return d.Krho / std::abs(maxDivv);
}

template<class Dataset, class... Ts>
void computeTimestep(size_t first, size_t last, Dataset& d, Ts... extraTimesteps)
{
    using T = typename Dataset::RealType;

    T minDtAcc = (d.g != 0.0) ? accelerationTimestep(first, last, d) : INFINITY;

    T minDtLoc = std::min({minDtAcc, d.minDtCourant, d.minDtRho, d.maxDtIncrease * d.minDt, extraTimesteps...});

    T minDtGlobal;
    MPI_Allreduce(&minDtLoc, &minDtGlobal, 1, MpiType<T>{}, MPI_MIN, MPI_COMM_WORLD);

    d.ttot += minDtGlobal;

    d.minDt_m1 = d.minDt;
    d.minDt    = minDtGlobal;
}

} // namespace sph
