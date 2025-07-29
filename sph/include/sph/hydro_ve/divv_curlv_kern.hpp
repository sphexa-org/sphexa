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
 * @brief Divergence of velocity vector field
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

template<size_t stride = 1, typename Tc, class T>
HOST_DEVICE_FUN inline void
divV_curlVJLoop(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box, const cstone::LocalIndex* neighbors,
                unsigned neighborsCount, const Tc* x, const Tc* y, const Tc* z, const T* vx, const T* vy, const T* vz,
                const T* h, const T* c11, const T* c12, const T* c13, const T* c22, const T* c23, const T* c33,
                const T* wh, const T* /*whd*/, const T* kx, const T* xm, T* divv, T* curlv, T* dV11, T* dV12, T* dV13,
                T* dV22, T* dV23, T* dV33, bool doGradV)
{
    auto xi  = x[i];
    auto yi  = y[i];
    auto zi  = z[i];
    auto vxi = vx[i];
    auto vyi = vy[i];
    auto vzi = vz[i];
    auto hi  = h[i];
    auto kxi = kx[i];

    auto hiInv  = T(1) / hi;
    auto hiInv3 = hiInv * hiInv * hiInv;

    // the 3 components of these vectors will be the derivatives in x,y,z directions
    cstone::Vec3<T> dVxi{0., 0., 0.}, dVyi{0., 0., 0.}, dVzi{0., 0., 0.};

    auto c11i = c11[i];
    auto c12i = c12[i];
    auto c13i = c13[i];
    auto c22i = c22[i];
    auto c23i = c23[i];
    auto c33i = c33[i];

    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[stride * pj];

        T rx = xi - x[j];
        T ry = yi - y[j];
        T rz = zi - z[j];

        applyPBC(box, T(2) * hi, rx, ry, rz);

        T r2   = rx * rx + ry * ry + rz * rz;
        T dist = std::sqrt(r2);

        T vx_ji = vx[j] - vxi;
        T vy_ji = vy[j] - vyi;
        T vz_ji = vz[j] - vzi;

        T v1 = dist * hiInv;
        T Wi = lt::lookup(wh, v1);

        cstone::Vec3<T> termA;
        termA[0] = -(c11i * rx + c12i * ry + c13i * rz) * Wi;
        termA[1] = -(c12i * rx + c22i * ry + c23i * rz) * Wi;
        termA[2] = -(c13i * rx + c23i * ry + c33i * rz) * Wi;

        T xmassj = xm[j];

        dVxi += (vx_ji * xmassj) * termA;
        dVyi += (vy_ji * xmassj) * termA;
        dVzi += (vz_ji * xmassj) * termA;
    }

    T norm_kxi = K * hiInv3 / kxi;
    divv[i]    = norm_kxi * (dVxi[0] + dVyi[1] + dVzi[2]);

    if (curlv != nullptr)
    {
        cstone::Vec3<T> curlV{dVzi[1] - dVyi[2], dVxi[2] - dVzi[0], dVyi[0] - dVxi[1]};
        curlv[i] = norm_kxi * std::sqrt(norm2(curlV));
    }

    if (doGradV)
    {
        dV11[i] = norm_kxi * dVxi[0];
        dV12[i] = norm_kxi * (dVxi[1] + dVyi[0]);
        dV13[i] = norm_kxi * (dVxi[2] + dVzi[0]);
        dV22[i] = norm_kxi * dVyi[1];
        dV23[i] = norm_kxi * (dVyi[2] + dVzi[1]);
        dV33[i] = norm_kxi * dVzi[2];
    }
}

} // namespace sph
