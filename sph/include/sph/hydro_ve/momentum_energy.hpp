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

template<bool avClean, class T, class Dataset>
void computeMomentumEnergy(const GroupView& groups, float* groupDt, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        gpu::computeMomentumEnergy<avClean>(groups, groupDt, d, box);
    }
    else
    {
        momentumAndEnergyIjLoop<avClean>(
            d.neighborhood, d.K, d.Kcour, d.Atmin, d.Atmax, d.ramp, d.vx.data(), d.vy.data(), d.vz.data(), d.m.data(),
            d.c.data(), d.kx.data(), d.alpha.data(), d.xm.data(), d.prho.data(), d.c11.data(), d.c12.data(),
            d.c13.data(), d.c22.data(), d.c23.data(), d.c33.data(), d.nc.data(), d.dV11.data(), d.dV12.data(),
            d.dV13.data(), d.dV22.data(), d.dV23.data(), d.dV33.data(), d.tdpdTrho.data(), d.wh.data(), d.du.data(),
            d.ax.data(), d.ay.data(), d.az.data(), d.dtCourant.data());

        auto minDt = std::numeric_limits<typename Dataset::HydroType>::infinity();
#pragma omp parallel for reduction(min : minDt)
        for (auto i = groups.firstBody; i < groups.lastBody; ++i)
            minDt = std::min(minDt, d.dtCourant[i]);
        d.minDtCourant = minDt;
    }
}

} // namespace sph
