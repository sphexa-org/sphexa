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
 * @brief Gradient of pressure and energy i-loop driver
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 */

#pragma once

#include "sph/sph_gpu.hpp"
#include "momentum_energy_kern.hpp"

namespace sph
{

template<bool avClean, class Tc, class Dataset>
void computeMomentumEnergyImpl(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<Tc>& box)
{
    using T = typename Dataset::HydroType;

    const cstone::LocalIndex* neighbors      = d.neighbors.data();
    const unsigned*           neighborsCount = d.nc.data();

    const auto* h        = d.h.data();
    const auto* m        = d.m.data();
    const auto* x        = d.x.data();
    const auto* y        = d.y.data();
    const auto* z        = d.z.data();
    const auto* vx       = d.vx.data();
    const auto* vy       = d.vy.data();
    const auto* vz       = d.vz.data();
    const auto* c        = d.c.data();
    const auto* prho     = d.prho.data();
    const auto* tdpdTrho = d.tdpdTrho.data();
    const auto* alpha    = d.alpha.data();

    const auto* c11 = d.c11.data();
    const auto* c12 = d.c12.data();
    const auto* c13 = d.c13.data();
    const auto* c22 = d.c22.data();
    const auto* c23 = d.c23.data();
    const auto* c33 = d.c33.data();

    const auto* dV11 = d.dV11.data();
    const auto* dV12 = d.dV12.data();
    const auto* dV13 = d.dV13.data();
    const auto* dV22 = d.dV22.data();
    const auto* dV23 = d.dV23.data();
    const auto* dV33 = d.dV33.data();

    auto* du       = d.du.data();
    auto* grad_P_x = d.ax.data();
    auto* grad_P_y = d.ay.data();
    auto* grad_P_z = d.az.data();

    const auto* wh  = d.wh.data();
    const auto* whd = d.whd.data();
    const auto* kx  = d.kx.data();
    const auto* xm  = d.xm.data();

    T minDt = INFINITY;

#pragma omp parallel for schedule(static) reduction(min : minDt)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        size_t   ni       = i - startIndex;
        unsigned ncCapped = stl::min(neighborsCount[i] - 1, d.ngmax);

        T maxvsignal = 0;

        momentumAndEnergyJLoop<avClean>(i, d.K, box, neighbors + d.ngmax * ni, ncCapped, x, y, z, vx, vy, vz, h, m,
                                        prho, tdpdTrho, c, c11, c12, c13, c22, c23, c33, d.Atmin, d.Atmax, d.ramp, wh,
                                        kx, xm, alpha, dV11, dV12, dV13, dV22, dV23, dV33, grad_P_x, grad_P_y, grad_P_z,
                                        du, &maxvsignal);

        T dt_i = tsKCourant(maxvsignal, h[i], c[i], d.Kcour);
        minDt  = std::min(minDt, dt_i);
    }

    d.minDtCourant = minDt;
}

template<bool avClean, class T, class Dataset>
void computeMomentumEnergy(const GroupView& grp, float* groupDt, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        cuda::computeMomentumEnergy<avClean>(grp, groupDt, d, box);
    }
    else { computeMomentumEnergyImpl<avClean>(grp.firstBody, grp.lastBody, d, box); }
}

} // namespace sph
