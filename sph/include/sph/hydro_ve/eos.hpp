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

#include "sph/sph_gpu.hpp"
#include "sph/eos.hpp"

namespace sph
{

/*! @brief ideal gas EOS interface w/o temperature for SPH where rho is computed on-the-fly
 *
 * @tparam Dataset
 * @param startIndex  index of first locally owned particle
 * @param endIndex    index of last locally owned particle
 * @param d           the dataset with the particle buffers
 *
 * In this simple version of equation of state, we calculate all dependent quantities
 * also for halos, not just assigned particles in [startIndex:endIndex], so that
 * we could potentially avoid halo exchange of p and c in return for exchanging halos of u.
 */
template<typename Dataset>
void computeEOS_Impl(size_t startIndex, size_t endIndex, Dataset& d)
{
    const auto* temp  = d.temp.data();
    const auto* m     = d.m.data();
    const auto* kx    = d.kx.data();
    const auto* xm    = d.xm.data();
    const auto* gradh = d.gradh.data();

    auto* prho = d.prho.data();
    auto* c    = d.c.data();

    bool storeRho = (d.rho.size() == d.m.size());
    bool storeP   = (d.p.size() == d.m.size());

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        auto rho      = kx[i] * m[i] / xm[i];
        auto [pi, ci] = idealGasEOS(temp[i], rho, d.muiConst, d.gamma);
        prho[i]       = pi / (kx[i] * m[i] * m[i] * gradh[i]);
        c[i]          = ci;
        if (storeRho) { d.rho[i] = rho; }
        if (storeP) { d.p[i] = pi; }
    }
}

template<class Dataset>
void computeEOS(size_t startIndex, size_t endIndex, Dataset& d)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        cuda::computeEOS(startIndex, endIndex, d.muiConst, d.gamma, rawPtr(d.devData.temp), rawPtr(d.devData.m),
                         rawPtr(d.devData.kx), rawPtr(d.devData.xm), rawPtr(d.devData.gradh), rawPtr(d.devData.prho),
                         rawPtr(d.devData.c), rawPtr(d.devData.rho), rawPtr(d.devData.p));
    }
    else { computeEOS_Impl(startIndex, endIndex, d); }
}

template<typename Dataset>
void computeIsothermalEOS_Impl(size_t startIndex, size_t endIndex, Dataset& d)
{
    const auto* m     = d.m.data();
    const auto* kx    = d.kx.data();
    const auto* xm    = d.xm.data();
    const auto* gradh = d.gradh.data();

    auto* prho = d.prho.data();
    auto* c    = d.c.data();
    auto* u    = d.u.data();

    bool storeRho = (d.rho.size() == d.m.size());
    bool storeP   = (d.p.size() == d.m.size());
    bool storeU   = (d.u.size() == d.m.size());

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        auto rho = kx[i] * m[i] / xm[i];
        auto pi  = isothermalEOS(c[i], rho);
        prho[i]  = pi / (kx[i] * m[i] * m[i] * gradh[i]);
        if (storeRho) { d.rho[i] = rho; }
        if (storeP) { d.p[i] = pi; }
        if (storeU) { u[i] = c[i] * c[i] / (d.gamma * (d.gamma - 1.)); }
    }
}

template<class Dataset>
void computeIsothermalEOS(size_t startIndex, size_t endIndex, Dataset& d)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        cuda::computeIsothermalEOS(startIndex, endIndex, d);
    }
    else { computeIsothermalEOS_Impl(startIndex, endIndex, d); }
}

template<class Dataset>
void computeIsothermalEosStd(size_t startIndex, size_t endIndex, Dataset& d)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        cuda::computeIsothermalEosStd(startIndex, endIndex, d);
    }
    else { throw std::runtime_error("computeIosthermalEosStd not implemented on CPU\n"); }
}

} // namespace sph
