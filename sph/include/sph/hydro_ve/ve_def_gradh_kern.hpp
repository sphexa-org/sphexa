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
 * @brief Volume definition and gradient of h architecture portable kernel
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 */

#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/sfc/box.hpp"

#include "sph/kernels.hpp"
#include "sph/table_lookup.hpp"

namespace sph
{

template<size_t stride = 1, class Tc, class Tm, class T>
HOST_DEVICE_FUN inline util::tuple<T, T> veDefGradhJLoop(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box,
                                                         const cstone::LocalIndex* neighbors, unsigned neighborsCount,
                                                         const Tc* x, const Tc* y, const Tc* z, const T* h, const Tm* m,
                                                         const T* wh, const T* whd, const T* xm)
{
    auto xi     = x[i];
    auto yi     = y[i];
    auto zi     = z[i];
    auto hi     = h[i];
    auto mi     = m[i];
    auto xmassi = xm[i];

    auto hInv  = T(1) / hi;
    auto h3Inv = hInv * hInv * hInv;

    // initialize with self-contribution
    auto kxi      = xmassi;
    auto whomegai = -T(3) * xmassi;
    auto wrho0i   = -T(3) * mi;

    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[stride * pj];

        T dist   = distancePBC(box, hi, xi, yi, zi, x[j], y[j], z[j]);
        T vloc   = dist * hInv;
        T w      = lt::lookup(wh, vloc);
        T dw     = lt::lookup(whd, vloc);
        T dterh  = -(T(3) * w + vloc * dw);
        T xmassj = xm[j];

        kxi += w * xmassj;
        whomegai += dterh * xmassj;
        wrho0i += dterh * m[j];
    }

    kxi *= K * h3Inv;
    whomegai *= K * h3Inv * hInv;
    wrho0i *= K * h3Inv * hInv;

    whomegai = whomegai * mi / xmassi + (kxi - K * xmassi * h3Inv) * wrho0i;
    T rhoi   = kxi * mi / xmassi;
    T dhdrho = -hi / (rhoi * T(3)); // This /3 is the dimension hard-coded.

    T gradhi = T(1);// - dhdrho * whomegai;
    return {kxi, gradhi};
}

} // namespace sph
