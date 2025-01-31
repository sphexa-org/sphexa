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
 * @brief Density i-loop OpenMP driver
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/cuda/cuda_utils.hpp"
#include "sph/sph_gpu.hpp"
#include "sph/particles_data_stubs.hpp"
#include "sph/eos.hpp"

namespace sph
{

/*! @brief Ideal gas EOS interface w/o temperature for SPH where rho is stored
 *
 * @tparam Dataset
 * @param startIndex  index of first locally owned particle
 * @param endIndex    index of last locally owned particle
 * @param d           the dataset with the particle buffers
 *
 * In this simple version of state equation, we calculate all depended quantities
 * also for halos, not just assigned particles in [startIndex:endIndex], so that
 * we could potentially avoid halo exchange of p and c in return for exchanging halos of u.
 */
template<typename Dataset>
void computeIdealGasEOS_HydroStd_Impl(size_t startIndex, size_t endIndex, Dataset& d)
{
    const auto* u    = d.u.data();
    const auto* temp = d.temp.data();
    const auto* rho  = d.rho.data();

    auto* p = d.p.data();
    auto* c = d.c.data();

    if (d.u.size() == 0)
    {
#pragma omp parallel for schedule(static)
        for (size_t i = startIndex; i < endIndex; ++i)
        {
            std::tie(p[i], c[i]) = idealGasEOS(temp[i], rho[i], d.muiConst, d.gamma);
        }
    }
    else
    {
#pragma omp parallel for schedule(static)
        for (size_t i = startIndex; i < endIndex; ++i)
        {
            std::tie(p[i], c[i]) = idealGasEOS_u(u[i], rho[i], d.gamma);
        }
    }
}

template<typename Dataset>
void computeIsothermalEOS_HydroStd_Impl(size_t startIndex, size_t endIndex, Dataset& d)
{
    const auto* rho = d.rho.data();

    auto* p      = d.p.data();
    auto* temp   = d.temp.data();
    auto  cConst = d.soundSpeedConst;

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        p[i] = isothermalEOS(cConst, rho[i]);
        if (temp) { temp[i] = 0; }
    }
}

template<typename Dataset>
void computePolytropicEOS_HydroStd_Impl(size_t startIndex, size_t endIndex, Dataset& d)
{
    const auto* rho = d.rho.data();

    auto* p    = d.p.data();
    auto* temp = d.temp.data();
    auto* c    = d.c.data();

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        std::tie(p[i], c[i]) = polytropicEOS(d.polytropic_const, d.polytropic_index, rho[i]);
        if (temp) { temp[i] = 0; }
    }
}

template<class Dataset>
void computeIdealGasEOS_HydroStd(size_t startIndex, size_t endIndex, Dataset& d)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        cuda::computeIdealGasEOS_HydroStd(startIndex, endIndex, d);
    }
    else { computeIdealGasEOS_HydroStd_Impl(startIndex, endIndex, d); }
}

template<class Dataset>
void computeIsothermalEOS_HydroStd(size_t startIndex, size_t endIndex, Dataset& d)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        cuda::computeIsothermalEOS_HydroStd(startIndex, endIndex, d);
    }
    else { computeIsothermalEOS_HydroStd_Impl(startIndex, endIndex, d); }
}

template<class Dataset>
void computePolytropicEOS_HydroStd(size_t startIndex, size_t endIndex, Dataset& d)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        cuda::computePolytropicEOS_HydroStd(startIndex, endIndex, d);
    }
    else { computePolytropicEOS_HydroStd_Impl(startIndex, endIndex, d); }
}

template<class Dataset>
void computeEOS_HydroStd(size_t startIndex, size_t endIndex, Dataset& d)
{
    if (d.eosChoice == EosType::idealGas) { computeIdealGasEOS_HydroStd(startIndex, endIndex, d); }
    else if (d.eosChoice == EosType::isothermal) { computeIsothermalEOS_HydroStd(startIndex, endIndex, d); }
    else if (d.eosChoice == EosType::polytropic) { computePolytropicEOS_HydroStd(startIndex, endIndex, d); }
}

} // namespace sph
