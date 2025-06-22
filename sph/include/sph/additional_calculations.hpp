/*
 * MIT License
 *
 * Copyright (c) 2023 CSCS, ETH Zurich
 *               2023 University of Basel
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
 * @brief additional fields i-loop driver
 * for calculating additional fields which require a neighbor traversal
 *
 * @author Lukas Schmidt
 */

#pragma once

#include "additional_calculations_kern.hpp"
#include "sph_gpu.hpp"

namespace sph
{

template<class Dataset>
void computeMarkRampImpl(size_t first, size_t last, Dataset& d)
{
    const cstone::LocalIndex* neighbors      = d.neighbors.data();
    const unsigned*           neighborsCount = d.nc.data();

    const auto* kx       = d.kx.data();
    const auto* xm       = d.xm.data();
    const auto* m        = d.m.data();
    auto*       markRamp = d.markRamp.data();

#pragma omp parallel for schedule(static)
    for (size_t i = first; i < last; ++i)
    {
        size_t   ni       = i - first;
        unsigned ncCapped = stl::min(neighborsCount[i] - 1, d.ngmax);
        markRampJLoop(i, neighbors + d.ngmax * ni, ncCapped, d.Atmin, d.Atmax, d.ramp, kx, xm, m, markRamp);
    }
}

template<class T, class Dataset>
void computeMarkRamp(const GroupView& grp, Dataset& d, const cstone::Box<T>& box)
{

    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{}) { gpu::computeMarkRamp(grp, d, box); }
    else { computeMarkRampImpl(grp.firstBody, grp.lastBody, d); }
}

template<class T, class Dataset>
void artificialGravity(size_t startIndex, size_t endIndex, Dataset& d, T grav)
{

    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        gpu::artificialGravity(startIndex, endIndex, d, grav);
    }
    else
    {
#pragma omp parallel for schedule(static)
        for (size_t i = startIndex; i < endIndex; i++)
        {
            d.ay[i] -= grav;
        }
    }
}

}; // namespace sph
