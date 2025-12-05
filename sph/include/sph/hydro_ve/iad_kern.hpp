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
 * @brief "Integral approach to derivative (IAD) implementation"
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

template<size_t stride = 1, class Tc, class T>
HOST_DEVICE_FUN inline void IADJLoop(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box,
                                     const cstone::LocalIndex* neighbors, unsigned neighborsCount, const Tc* x,
                                     const Tc* y, const Tc* z, const T* h, const T* wh, const T* /*whd*/, const T* xm,
                                     const T* kx, T* c11, T* c12, T* c13, T* c22, T* c23, T* c33)
{
    T tau11 = 0.0, tau12 = 0.0, tau13 = 0.0, tau22 = 0.0, tau23 = 0.0, tau33 = 0.0;

    auto xi = x[i];
    auto yi = y[i];
    auto zi = z[i];

    auto hi    = h[i];
    auto hiInv = T(1) / hi;

    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[stride * pj];

        T rx = (xi - x[j]);
        T ry = (yi - y[j]);
        T rz = (zi - z[j]);

        applyPBC(box, T(2) * hi, rx, ry, rz);

        T dist = std::sqrt(rx * rx + ry * ry + rz * rz);

        // calculate the v as ratio between the distance and the smoothing length
        T vloc = dist * hiInv;
        T w    = lt::lookup(wh, vloc);

        T volj_w = xm[j] / kx[j] * w;

        tau11 += rx * rx * volj_w;
        tau12 += rx * ry * volj_w;
        tau13 += rx * rz * volj_w;
        tau22 += ry * ry * volj_w;
        tau23 += ry * rz * volj_w;
        tau33 += rz * rz * volj_w;
    }

    auto getExp    = [](T val) { return (val == T(0) ? 0 : std::ilogb(val)); };
    int  tauExpSum = getExp(tau11) + getExp(tau12) + getExp(tau13) + getExp(tau22) + getExp(tau23) + getExp(tau33);
    // normalize with 2^-averageTauExponent, ldexp(a, b) == a * 2^b
    T normalization = std::ldexp(T(1), -tauExpSum / 6);

    tau11 *= normalization;
    tau12 *= normalization;
    tau13 *= normalization;
    tau22 *= normalization;
    tau23 *= normalization;
    tau33 *= normalization;

    T det = tau11 * tau22 * tau33 + T(2) * tau12 * tau23 * tau13 - tau11 * tau23 * tau23 - tau22 * tau13 * tau13 -
            tau33 * tau12 * tau12;

    // note normalization factor: cij have units of 1/tau because det is proportional to tau^3, so we have to
    // divide by K/h^3
    T factor = normalization * (hi * hi * hi) / (det * K);
    if (std::isnan(factor) && neighborsCount == 0) { factor = T(0); }

    c11[i] = (tau22 * tau33 - tau23 * tau23) * factor;
    c12[i] = (tau13 * tau23 - tau33 * tau12) * factor;
    c13[i] = (tau12 * tau23 - tau22 * tau13) * factor;
    c22[i] = (tau11 * tau33 - tau13 * tau13) * factor;
    c23[i] = (tau13 * tau12 - tau11 * tau23) * factor;
    c33[i] = (tau11 * tau22 - tau12 * tau12) * factor;
}

} // namespace sph
