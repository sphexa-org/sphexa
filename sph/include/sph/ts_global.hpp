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

#include "acceleration_timestep_gpu.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/util/array.hpp"
#include "kernels.hpp"

namespace sph
{

//! @brief limit time-step based on accelerations when gravity is enabled
//! Computes etaAcc * min(sqrt(h[i] / norm(a[i])))
template<class Dataset>
auto accelerationTimestep(size_t first, size_t last, const Dataset& d)
{
    using T = typename Dataset::RealType;
    if (last <= first) return std::numeric_limits<T>::infinity();

    //! @brief minimum value of all {h_i^2 / a_i^2}
    T minH2_A2 = std::numeric_limits<T>::infinity();
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        minH2_A2 = accelerationTimestepGPU(first, last, rawPtr(d.devData.ax), rawPtr(d.devData.ay),
                                           rawPtr(d.devData.az), rawPtr(d.devData.h));
    }
    else
    {
#pragma omp parallel for reduction(min : minH2_A2)
        for (size_t i = first; i < last; ++i)
        {
            cstone::Vec3<T> A{d.ax[i], d.ay[i], d.az[i]};
            minH2_A2 = std::min(minH2_A2, d.h[i] * d.h[i] / norm2(A));
        }
    }

    return d.etaAcc * std::pow(minH2_A2, 0.25);
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

    util::array<T, 4> varsIn{minDtLoc, 0, 0, -T(d.accSize())}, varsOut;
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        varsIn[1] = -int(d.devData.stackUsedNc);
        varsIn[2] = -int(d.devData.stackUsedGravity);
    }
    MPI_Allreduce(varsIn.data(), varsOut.data(), varsIn.size(), MpiType<T>{}, MPI_MIN, MPI_COMM_WORLD);
    T minDtGlobal = varsOut[0];
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        d.devData.stackUsedNc      = int(-varsOut[1]);
        d.devData.stackUsedGravity = int(-varsOut[2]);
    }
    d.maxHalos = int(-varsOut[3]);

    d.ttot += minDtGlobal;

    d.minDt_m1 = d.minDt;
    d.minDt    = minDtGlobal;
}

} // namespace sph
