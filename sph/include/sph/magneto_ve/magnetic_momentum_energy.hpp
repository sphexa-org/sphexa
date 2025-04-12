/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUTh WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUTh NOTh LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENTh SHALL THE
 * AUTHORS OR COPYRIGHTh HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORTh OR OTHERWISE, ARISING FROM,
 * OUTh OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file calculates the momentum and energy with the magnetic stress tensor
 *
 * @author Lukas Schmidt
 *
 */

#pragma once

#include "sph/sph_gpu.hpp"
#include "magnetic_momentum_energy_kern.hpp"

namespace sph::magneto
{
template<bool avClean, class Tc, class SimData>
void computeMagneticMomentumEnergyImpl(size_t startIndex, size_t endIndex, SimData& sim, const cstone::Box<Tc>& box)
{
    using T = typename SimData::HydroType;

    auto& d  = sim.hydro;
    auto& md = sim.magneto;

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
    const auto* p        = d.p.data();
    const auto* tdpdTrho = d.tdpdTrho.data();
    const auto* alpha    = d.alpha.data();
    const auto* gradh    = d.gradh.data();

    const auto* c11 = d.c11.data();
    const auto* c12 = d.c12.data();
    const auto* c13 = d.c13.data();
    const auto* c22 = d.c22.data();
    const auto* c23 = d.c23.data();
    const auto* c33 = d.c33.data();

    const auto* dvxdx = md.dvxdx.data();
    const auto* dvxdy = md.dvxdy.data();
    const auto* dvxdz = md.dvxdz.data();
    const auto* dvydx = md.dvydx.data();
    const auto* dvydy = md.dvydy.data();
    const auto* dvydz = md.dvydz.data();
    const auto* dvzdx = md.dvzdx.data();
    const auto* dvzdy = md.dvzdy.data();
    const auto* dvzdz = md.dvzdz.data();

    const auto* Bx = md.Bx.data();
    const auto* By = md.By.data();
    const auto* Bz = md.Bz.data();

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

        magneticMomentumJLoop<avClean>(i, d.K, md.mu_0, box, neighbors + d.ngmax * ni, ncCapped, x, y, z, vx, vy, vz, h,
                                       m, p, tdpdTrho, c, c11, c12, c13, c22, c23, c33, d.Atmin, d.Atmax, d.ramp, wh,
                                       kx, xm, alpha, dvxdx, dvxdy, dvxdz, dvydx, dvydy, dvydz, dvzdx, dvzdy, dvzdz, Bx,
                                       By, Bz, gradh, grad_P_x, grad_P_y, grad_P_z, du, &maxvsignal);

        T rhoi            = kx[i] * m[i] / xm[i];
        T v_alfven2       = (Bx[i] * Bx[i] + By[i] * By[i] + Bz[i] * Bz[i]) / (md.mu_0 * rhoi);
        T magneticVsignal = std::sqrt(c[i] * c[i] + v_alfven2);

        T dt_i = maxvsignal > T(0) ? std::min(d.Kcour * h[i] / magneticVsignal, d.Kcour * h[i] / maxvsignal)
                                   : d.Kcour * h[i] / magneticVsignal;

        // T dt_i = tsKCourant(maxvsignal, h[i], magneticVsignal, d.Kcour);
        minDt = std::min(minDt, dt_i);
    }

    d.minDtCourant = minDt;
}

template<bool avClean, class T, class SimData>
void computeMomentumEnergy(const GroupView& grp, float* groupDt, SimData& sim, const cstone::Box<T>& box)
{
    if constexpr (cstone::HaveGpu<typename SimData::AcceleratorType>{})
    {
        cuda::computeMagneticMomentumEnergy<avClean>(grp, groupDt, sim.hydro, sim.magneto, box);
    }
    else { computeMagneticMomentumEnergyImpl<avClean>(grp.firstBody, grp.lastBody, sim, box); }
}

} // namespace sph::magneto
