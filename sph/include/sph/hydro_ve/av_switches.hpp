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
void computeAVswitches(const GroupView& grp, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{}) { gpu::computeAVswitches(grp, d, box); }
    else
    {
        // caution: d.alpha is used as input and output, this fails on neighborhoods that exploit symmetry!
        AVswitchesIjLoop(d.neighborhood, d.K, d.minDt, d.alphamin, d.alphamax, d.decay_constant, d.xm.data(),
                         d.kx.data(), d.divv.data(), d.alpha.data(), d.vx.data(), d.vy.data(), d.vz.data(), d.c.data(),
                         d.c11.data(), d.c12.data(), d.c13.data(), d.c22.data(), d.c23.data(), d.c33.data(),
                         d.wh.data(), d.alpha.data());
    }
}

} // namespace sph
