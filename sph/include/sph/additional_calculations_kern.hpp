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
 * @brief additional fields kernel
 *
 * @author Lukas Schmidt
 */

#include "cstone/cuda/annotation.hpp"

namespace sph
{

template<size_t stride = 1, class T, class Tm>
HOST_DEVICE_FUN void markRampJLoop(cstone::LocalIndex i, const cstone::LocalIndex* neighbors, unsigned neighborsCount,
                                   T Atmin, T Atmax, T ramp, const T* kx, const T* xm, const Tm* m, T* markRamp)
{
    auto rhoi   = kx[i] * m[i] / xm[i];
    markRamp[i] = T(0);

    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j    = neighbors[stride * pj];
        auto               rhoj = kx[j] * m[j] / xm[j];

        T Atwood = (std::abs(rhoi - rhoj)) / (rhoi + rhoj);
        if (Atwood > Atmax) { markRamp[i] += T(1); }
        else if (Atwood >= Atmin)
        {
            T sigma_ij = ramp * (Atwood - Atmin);
            markRamp[i] += sigma_ij;
        }
    }
    markRamp[i] /= neighborsCount;
}

} // namespace sph
