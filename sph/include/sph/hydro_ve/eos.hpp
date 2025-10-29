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
#include "sph/helmholtz_eos.hpp"

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
void computeIdealGasEOS_Impl(size_t startIndex, size_t endIndex, Dataset& d)
{
    const auto* u     = d.u.data();
    const auto* temp  = d.temp.data();
    const auto* m     = d.m.data();
    const auto* kx    = d.kx.data();
    const auto* xm    = d.xm.data();
    const auto* gradh = d.gradh.data();

    auto* prho = d.prho.data();
    auto* c    = d.c.data();

    bool storeRho = (d.rho.size() == d.m.size());
    bool storeP   = (d.p.size() == d.m.size());

    if (d.u.empty())
    {
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
    else
    {
#pragma omp parallel for schedule(static)
        for (size_t i = startIndex; i < endIndex; ++i)
        {
            auto rho      = kx[i] * m[i] / xm[i];
            auto [pi, ci] = idealGasEOS_u(u[i], rho, d.gamma);
            prho[i]       = pi / (kx[i] * m[i] * m[i] * gradh[i]);
            c[i]          = ci;
            if (storeRho) { d.rho[i] = rho; }
            if (storeP) { d.p[i] = pi; }
        }
    }
}

template<typename Dataset>
void computeIsothermalEOS_Impl(size_t startIndex, size_t endIndex, Dataset& d)
{
    const auto* m     = d.m.data();
    const auto* kx    = d.kx.data();
    const auto* xm    = d.xm.data();
    const auto* gradh = d.gradh.data();

    auto  cConst = d.soundSpeedConst;
    auto* c      = d.c.data();
    auto* prho   = d.prho.data();
    auto* temp   = d.temp.data();

    bool storeRho = (d.rho.size() == d.m.size());
    bool storeP   = (d.p.size() == d.m.size());

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        auto rho = kx[i] * m[i] / xm[i];
        auto pi  = isothermalEOS(cConst, rho);
        prho[i]  = pi / (kx[i] * m[i] * m[i] * gradh[i]);
        c[i]     = cConst; // c is used in AV-switches and momentum energy, need to set correct constant value
        if (storeRho) { d.rho[i] = rho; }
        if (storeP) { d.p[i] = pi; }
        if (temp) { temp[i] = 0; }
    }
}

template<typename Dataset>
void computePolytropicEOS_Impl(size_t startIndex, size_t endIndex, Dataset& d)
{
    const auto* m     = d.m.data();
    const auto* kx    = d.kx.data();
    const auto* xm    = d.xm.data();
    const auto* gradh = d.gradh.data();

    auto* prho = d.prho.data();
    auto* temp = d.temp.data();
    auto* c    = d.c.data();

    bool storeRho = (d.rho.size() == d.m.size());
    bool storeP   = (d.p.size() == d.m.size());

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        auto rho      = kx[i] * m[i] / xm[i];
        auto [pi, ci] = polytropicEOS(d.polytropic_const, d.polytropic_index, rho);
        prho[i]       = pi / (kx[i] * m[i] * m[i] * gradh[i]);
        c[i]          = ci;
        if (storeRho) { d.rho[i] = rho; }
        if (storeP) { d.p[i] = pi; }
        if (temp) { temp[i] = 0; }
    }
}

template<typename Dataset>
void computeHelmholtzEOS_Impl(size_t startIndex, size_t endIndex, Dataset& d)
{
    const auto* kx    = d.kx.data();
    const auto* xm    = d.xm.data();
    const auto* m     = d.m.data();
    const auto* temp  = d.temp.data();
    const auto* abar  = d.abar.data();
    const auto* zbar  = d.zbar.data();
    const auto* gradh = d.gradh.data();

    auto* prho     = d.prho.data();
    auto* c        = d.c.data();
    auto* cv       = d.cv.data();
    auto* tdpdTrho = d.tdpdTrho.data();
    auto* u        = d.u.data();

    bool storeRho      = (d.rho.size() == d.m.size());
    bool storeP        = (d.p.size() == d.m.size());
    bool storeCv       = (d.cv.size() == d.m.size());
    bool storeTdpdTrho = (d.tdpdTrho.size() == d.m.size());

    Helmholtz_EOS& helmEOS = sph::Helmholtz_EOS::instance();

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        auto rho = kx[i] * m[i] / xm[i];
        auto pi  = prho[i] / rho;
        // get dpdt instead of u and calculate tdpdtrho = temp * dp/dT * prho
        auto [dpdt, cvi] = helmEOS.helmholtzEOS(temp[i], rho, abar[i], zbar[i], &c[i], &pi);

        prho[i] = pi / (kx[i] * m[i] * m[i] * gradh[i]);
        // c[i]    = ci;
        if (storeRho) { d.rho[i] = rho; }
        if (storeTdpdTrho) { tdpdTrho[i] = temp[i] * dpdt * prho[i]; }
        // if (storeP) { d.p[i] = pi; }
        if (storeCv) { d.cv[i] = cvi; }
        // if (storeU) { d.u[i] = ui; }
    }
}

template<class Dataset>
void computeIdealGasEOS(size_t startIndex, size_t endIndex, Dataset& d)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        cuda::computeIdealGasEOS(startIndex, endIndex, d.muiConst, d.gamma, rawPtr(d.devData.temp), rawPtr(d.devData.u),
                                 rawPtr(d.devData.m), rawPtr(d.devData.kx), rawPtr(d.devData.xm),
                                 rawPtr(d.devData.gradh), rawPtr(d.devData.prho), rawPtr(d.devData.c),
                                 rawPtr(d.devData.rho), rawPtr(d.devData.p));
    }
    else
    {
        computeIdealGasEOS_Impl(startIndex, endIndex, d);
    }
}

template<class Dataset>
void computeIsothermalEOS(size_t startIndex, size_t endIndex, Dataset& d)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        cuda::computeIsothermalEOS(startIndex, endIndex, d.soundSpeedConst, rawPtr(d.devData.c), rawPtr(d.devData.rho),
                                   rawPtr(d.devData.p), rawPtr(d.devData.m), rawPtr(d.devData.kx), rawPtr(d.devData.xm),
                                   rawPtr(d.devData.gradh), rawPtr(d.devData.prho), rawPtr(d.devData.temp));
    }
    else
    {
        computeIsothermalEOS_Impl(startIndex, endIndex, d);
    }
}

template<class Dataset>
void computePolytropicEOS(size_t startIndex, size_t endIndex, Dataset& d)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        cuda::computePolytropicEOS(startIndex, endIndex, d.polytropic_const, d.polytropic_index, rawPtr(d.devData.rho),
                                   rawPtr(d.devData.p), rawPtr(d.devData.m), rawPtr(d.devData.kx), rawPtr(d.devData.xm),
                                   rawPtr(d.devData.gradh), rawPtr(d.devData.prho), rawPtr(d.devData.temp),
                                   rawPtr(d.devData.c));
    }
    else
    {
        computePolytropicEOS_Impl(startIndex, endIndex, d);
    }
}

template<class Dataset>
void computeHelmholtzEOS(size_t startIndex, size_t endIndex, Dataset& d)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        cuda::computeHelmholtzEOS(startIndex, endIndex, rawPtr(d.devData.kx), rawPtr(d.devData.xm), rawPtr(d.devData.m),
                                  rawPtr(d.devData.temp), rawPtr(d.devData.abar), rawPtr(d.devData.zbar),
                                  rawPtr(d.devData.gradh), rawPtr(d.devData.prho), rawPtr(d.devData.c),
                                  rawPtr(d.devData.tdpdTrho));
    }
    else
    {
        computeHelmholtzEOS_Impl(startIndex, endIndex, d);
    }
}

template<class Dataset>
void computeEOS(size_t startIndex, size_t endIndex, Dataset& d)
{
    if (d.eosChoice == EosType::idealGas) { computeIdealGasEOS(startIndex, endIndex, d); }
    else if (d.eosChoice == EosType::isothermal) { computeIsothermalEOS(startIndex, endIndex, d); }
    else if (d.eosChoice == EosType::polytropic) { computePolytropicEOS(startIndex, endIndex, d); }
    else if (d.eosChoice == EosType::helmholtz) { computeHelmholtzEOS(startIndex, endIndex, d); }
}

} // namespace sph
