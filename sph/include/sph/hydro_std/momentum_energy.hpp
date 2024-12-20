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
 * @brief Pressure gradient (momentum) and energy i-loop OpenMP driver
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "sph/sph_gpu.hpp"
#include "momentum_energy_kern.hpp"

namespace sph
{

template<class Tc, class Dataset>
void computeMomentumEnergyStdImpl(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<Tc>& box)
{
    using T = typename Dataset::HydroType;

    const cstone::LocalIndex* neighbors      = d.neighbors.data();
    const unsigned*           neighborsCount = d.nc.data();

    const auto* h   = d.h.data();
    const auto* m   = d.m.data();
    const auto* x   = d.x.data();
    const auto* y   = d.y.data();
    const auto* z   = d.z.data();
    const auto* vx  = d.vx.data();
    const auto* vy  = d.vy.data();
    const auto* vz  = d.vz.data();
    const auto* rho = d.rho.data();
    const auto* c   = d.c.data();
    const auto* p   = d.p.data();

    const auto* c11 = d.c11.data();
    const auto* c12 = d.c12.data();
    const auto* c13 = d.c13.data();
    const auto* c22 = d.c22.data();
    const auto* c23 = d.c23.data();
    const auto* c33 = d.c33.data();

    auto* du       = d.du.data();
    auto* grad_P_x = d.ax.data();
    auto* grad_P_y = d.ay.data();
    auto* grad_P_z = d.az.data();

    const auto* wh  = d.wh.data();
    const auto* whd = d.whd.data();

    T minDt = INFINITY;

#pragma omp parallel for schedule(static) reduction(min : minDt)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        size_t ni = i - startIndex;

        T maxvsignal = 0;

        unsigned ncCapped = std::min(neighborsCount[i] - 1, d.ngmax);
        momentumAndEnergyJLoop(i, d.K, box, neighbors + d.ngmax * ni, ncCapped, x, y, z, vx, vy, vz, h, m, rho, p, c,
                               c11, c12, c13, c22, c23, c33, wh, whd, grad_P_x, grad_P_y, grad_P_z, du, &maxvsignal);

        T dt_i = tsKCourant(maxvsignal, h[i], c[i], d.Kcour);
        minDt  = std::min(minDt, dt_i);
    }

    d.minDtCourant = minDt;
}

template<class T, class Dataset>
void computeMomentumEnergySTD(const GroupView& groups, float* groupDt, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        computeMomentumEnergyStdGpu(groups, groupDt, d, box);
    }
    else { computeMomentumEnergyStdImpl(groups.firstBody, groups.lastBody, d, box); }
}

} // namespace sph
