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
 * @brief Integral-approach-to-derivative i-loop OpenMP driver
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "sph/sph_gpu.hpp"
#include "iad_kern.hpp"

namespace sph
{

template<class T, class Dataset>
void computeIADImpl(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<T>& box)
{
    const cstone::LocalIndex* neighbors      = d.neighbors.data();
    const unsigned*           neighborsCount = d.nc.data();

    const auto* h   = d.h.data();
    const auto* m   = d.m.data();
    const auto* x   = d.x.data();
    const auto* y   = d.y.data();
    const auto* z   = d.z.data();
    const auto* rho = d.rho.data();

    auto* c11 = d.c11.data();
    auto* c12 = d.c12.data();
    auto* c13 = d.c13.data();
    auto* c22 = d.c22.data();
    auto* c23 = d.c23.data();
    auto* c33 = d.c33.data();

    const auto* wh  = d.wh.data();
    const auto* whd = d.whd.data();

#pragma omp parallel for schedule(static)
    for (cstone::LocalIndex i = startIndex; i < endIndex; ++i)
    {
        size_t   ni       = i - startIndex;
        unsigned ncCapped = std::min(neighborsCount[i] - 1, d.ngmax);
        IADJLoopSTD(i, d.K, box, neighbors + d.ngmax * ni, ncCapped, x, y, z, h, m, rho, wh, whd, c11, c12, c13, c22,
                    c23, c33);
    }
}

template<class T, class Dataset>
void computeIAD(const GroupView& groups, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{}) { computeIADGpu(groups, d, box); }
    else { computeIADImpl(groups.firstBody, groups.lastBody, d, box); }
}

} // namespace sph
