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

#include "cstone/traversal/ijloop/ijloop.hpp"

#include "sph/table_lookup.hpp"

namespace sph
{

template<class T>
struct VeDefGradHInteraction
{
    const T *wh, *whd;

    template<class ParticleData, class Tc>
    constexpr auto operator()(const ParticleData& iData, const ParticleData& jData, cstone::Vec3<Tc> const& /* r_ij */,
                              T r2) const
    {
        const auto [i, iPos, hi, mi, xmassi] = iData;
        const auto [j, jPos, hj, mj, xmassj] = jData;

        auto hInv = T(1) / hi;

        T dist  = std::sqrt(r2);
        T vloc  = dist * hInv;
        T w     = lt::lookup(wh, vloc);
        T dw    = lt::lookup(whd, vloc);
        T dterh = -(T(3) * w + vloc * dw);

        T kxi      = w * xmassj;
        T whomegai = dterh * xmassj;
        T wrho0i   = dterh * mj;

        return std::make_tuple(kxi, whomegai, wrho0i);
    }
};

template<class T, class Tc>
struct VeDefGradHPostamble
{
    Tc K;

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        const auto [i, iPos, hi, mi, xmassi] = iData;
        auto [kxi, whomegai, wrho0i]         = result;

        auto hInv  = T(1) / hi;
        auto h3Inv = hInv * hInv * hInv;

        kxi *= K * h3Inv;
        whomegai *= K * h3Inv * hInv;
        wrho0i *= K * h3Inv * hInv;

        whomegai = whomegai * mi / xmassi + (kxi - K * xmassi * h3Inv) * wrho0i;
        T rhoi   = kxi * mi / xmassi;
        T dhdrho = -hi / (rhoi * T(3)); // This /3 is the dimension hard-coded.

        T gradhi = T(1) - dhdrho * whomegai;
#ifndef NDEBUG
        if (std::isnan(rhoi))
        {
            printf("ERROR::Density(%zu) density %f, position: (%f %f %f), h: %f\n", size_t(i), rhoi, iPos[0], iPos[1],
                   iPos[2], hi);
        }
#endif
        return std::make_tuple(kxi, gradhi);
    }
};

template<class Neighbordhood, class Tc, class Tm, class T>
void veDefGradhIjLoop(const Neighbordhood& neighborhood, Tc K, const Tm* m, const T* xm, const T* wh, const T* whd,
                      T* kx, T* gradh)
{
    neighborhood.ijLoop(std::make_tuple(m, xm), std::make_tuple(kx, gradh), VeDefGradHInteraction{wh, whd},
                        VeDefGradHPostamble<T, Tc>{K});
}

} // namespace sph
