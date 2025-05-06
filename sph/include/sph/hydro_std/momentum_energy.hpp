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

template<class T, class Dataset>
void computeMomentumEnergySTD(const GroupView& groups, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{}) { computeMomentumEnergyStdGpu(groups, d, box); }
    else
    {
        momentumAndEnergyIjLoop(d.neighborhood, d.K, d.Kcour, d.m.data(), d.rho.data(), d.vx.data(), d.vy.data(),
                                d.vz.data(), d.p.data(), d.c.data(), d.c11.data(), d.c12.data(), d.c13.data(),
                                d.c22.data(), d.c23.data(), d.c33.data(), d.wh.data(), d.du.data(), d.ax.data(),
                                d.ay.data(), d.az.data(), d.dtCourant.data());

        auto minDt = std::numeric_limits<typename Dataset::HydroType>::infinity();
#pragma omp parallel for reduction(min : minDt)
        for (auto i = groups.firstBody; i < groups.lastBody; ++i)
            minDt = std::min(minDt, d.dtCourant[i]);
        d.minDtCourant = minDt;
    }
}

} // namespace sph
