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
 * @brief Artifical viscosity switches i-loop driver
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 */

#pragma once

#include "sph/sph_gpu.hpp"
#include "av_switches_kern.hpp"

namespace sph
{

template<class T, class Dataset>
void computeAVswitchesImpl(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<T>& box)
{
    const cstone::LocalIndex* neighbors      = d.neighbors.data();
    const unsigned*           neighborsCount = d.nc.data();

    const auto* x  = d.x.data();
    const auto* y  = d.y.data();
    const auto* z  = d.z.data();
    const auto* h  = d.h.data();
    const auto* vx = d.vx.data();
    const auto* vy = d.vy.data();
    const auto* vz = d.vz.data();
    const auto* c  = d.c.data();

    const auto* c11 = d.c11.data();
    const auto* c12 = d.c12.data();
    const auto* c13 = d.c13.data();
    const auto* c22 = d.c22.data();
    const auto* c23 = d.c23.data();
    const auto* c33 = d.c33.data();

    const auto* divv = d.divv.data();
    const auto* wh   = d.wh.data();
    const auto* whd  = d.whd.data();
    const auto* kx   = d.kx.data();
    const auto* xm   = d.xm.data();

    auto* alpha = d.alpha.data();

#pragma omp parallel for
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        size_t   ni       = i - startIndex;
        unsigned ncCapped = std::min(neighborsCount[i] - 1, d.ngmax);
        alpha[i] = AVswitchesJLoop(i, d.K, box, neighbors + d.ngmax * ni, ncCapped, x, y, z, vx, vy, vz, h, c, c11, c12,
                                   c13, c22, c23, c33, wh, whd, kx, xm, divv, d.minDt, d.alphamin, d.alphamax,
                                   d.decay_constant, alpha[i]);
    }
}

template<class T, class Dataset>
void computeAVswitches(const GroupView& grp, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{}) { cuda::computeAVswitches(grp, d, box); }
    else { computeAVswitchesImpl(grp.firstBody, grp.lastBody, d, box); }
}

} // namespace sph
