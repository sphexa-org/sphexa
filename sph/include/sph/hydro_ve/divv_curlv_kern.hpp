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

#include "cstone/traversal/ijloop/ijloop.hpp"

#include "sph/table_lookup.hpp"

namespace sph
{

template<class T>
struct DivVCurlVInteraction
{
    const T* wh;

    template<class ParticleData, class Tc>
    constexpr auto operator()(const ParticleData& iData, const ParticleData& jData, const cstone::Vec3<Tc>& r_ij,
                              const T r2) const
    {
        auto const [i, iPos, hi, vxi, vyi, vzi, xmi, kxi, c11i, c12i, c13i, c22i, c23i, c33i] = iData;
        auto const [j, jPos, hj, vxj, vyj, vzj, xmj, kxj, c11j, c12j, c13j, c22j, c23j, c33j] = jData;

        T hiInv = T(1) / hi;

        T rx   = r_ij[0];
        T ry   = r_ij[1];
        T rz   = r_ij[2];
        T dist = std::sqrt(r2);

        T vx_ji = vxj - vxi;
        T vy_ji = vyj - vyi;
        T vz_ji = vzj - vzi;

        T v1 = dist * hiInv;
        T Wi = lt::lookup(wh, v1);

        const T dVxiXFactor = (vx_ji * xmj) * (-rx * Wi);
        const T dVxiYFactor = (vx_ji * xmj) * (-ry * Wi);
        const T dVxiZFactor = (vx_ji * xmj) * (-rz * Wi);
        const T dVyiXFactor = (vy_ji * xmj) * (-rx * Wi);
        const T dVyiYFactor = (vy_ji * xmj) * (-ry * Wi);
        const T dVyiZFactor = (vy_ji * xmj) * (-rz * Wi);
        const T dVziXFactor = (vz_ji * xmj) * (-rx * Wi);
        const T dVziYFactor = (vz_ji * xmj) * (-ry * Wi);
        const T dVziZFactor = (vz_ji * xmj) * (-rz * Wi);
        return std::make_tuple(dVxiXFactor, dVxiYFactor, dVxiZFactor, //
                               dVyiXFactor, dVyiYFactor, dVyiZFactor, //
                               dVziXFactor, dVziYFactor, dVziZFactor);
    }
};

template<bool DoCurlV, bool DoGradV, class T, class Tc>
struct DivVCurlVPostamble
{
    Tc K;

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        auto const [i, iPos, hi, vxi, vyi, vzi, xmi, kxi, c11i, c12i, c13i, c22i, c23i, c33i] = iData;
        auto const [dVxiXFactor, dVxiYFactor, dVxiZFactor, dVyiXFactor, dVyiYFactor, dVyiZFactor, dVziXFactor,
                    dVziYFactor, dVziZFactor]                                                 = result;
        const cstone::Vec3<T> dVxi = {c11i * dVxiXFactor + c12i * dVxiYFactor + c13i * dVxiZFactor,
                                      c12i * dVxiXFactor + c22i * dVxiYFactor + c23i * dVxiZFactor,
                                      c13i * dVxiXFactor + c23i * dVxiYFactor + c33i * dVxiZFactor};
        const cstone::Vec3<T> dVyi = {c11i * dVyiXFactor + c12i * dVyiYFactor + c13i * dVyiZFactor,
                                      c12i * dVyiXFactor + c22i * dVyiYFactor + c23i * dVyiZFactor,
                                      c13i * dVyiXFactor + c23i * dVyiYFactor + c33i * dVyiZFactor};
        const cstone::Vec3<T> dVzi = {c11i * dVziXFactor + c12i * dVziYFactor + c13i * dVziZFactor,
                                      c12i * dVziXFactor + c22i * dVziYFactor + c23i * dVziZFactor,
                                      c13i * dVziXFactor + c23i * dVziYFactor + c33i * dVziZFactor};

        T hiInv  = T(1) / hi;
        T hiInv3 = hiInv * hiInv * hiInv;

        T norm_kxi = K * hiInv3 / kxi;
        T divvi    = norm_kxi * (dVxi[0] + dVyi[1] + dVzi[2]);

        cstone::Vec3<T> curlV{dVzi[1] - dVyi[2], dVxi[2] - dVzi[0], dVyi[0] - dVxi[1]};
        T               curlvi = norm_kxi * std::sqrt(norm2(curlV));

        T dV11i = norm_kxi * dVxi[0];
        T dV12i = norm_kxi * (dVxi[1] + dVyi[0]);
        T dV13i = norm_kxi * (dVxi[2] + dVzi[0]);
        T dV22i = norm_kxi * dVyi[1];
        T dV23i = norm_kxi * (dVyi[2] + dVzi[1]);
        T dV33i = norm_kxi * dVzi[2];

        if constexpr (DoCurlV && DoGradV)
            return std::make_tuple(divvi, curlvi, dV11i, dV12i, dV13i, dV22i, dV23i, dV33i);
        else if constexpr (DoCurlV)
            return std::make_tuple(divvi, curlvi);
        else if constexpr (DoGradV)
            return std::make_tuple(divvi, dV11i, dV12i, dV13i, dV22i, dV23i, dV33i);
        else
            return std::make_tuple(divvi);
    }
};

template<class Neighborhood, class Tc, class T>
void divVCurlVIjLoop(const Neighborhood& neighborhood, Tc K, const T* vx, const T* vy, const T* vz, const T* xm,
                     const T* kx, const T* c11, const T* c12, const T* c13, const T* c22, const T* c23, const T* c33,
                     const T* wh, T* divv, T* curlv, T* dV11, T* dV12, T* dV13, T* dV22, T* dV23, T* dV33, bool doGradV)
{
    const auto input = std::make_tuple(vx, vy, vz, xm, kx, c11, c12, c13, c22, c23, c33);
    if (curlv && doGradV)
    {
        const auto output = std::make_tuple(divv, curlv, dV11, dV12, dV13, dV22, dV23, dV33);
        neighborhood.ijLoop(input, output, DivVCurlVInteraction<T>{wh}, DivVCurlVPostamble<true, true, T, Tc>{K});
    }
    else if (curlv)
    {
        const auto output = std::make_tuple(divv, curlv);
        neighborhood.ijLoop(input, output, DivVCurlVInteraction<T>{wh}, DivVCurlVPostamble<true, false, T, Tc>{K});
    }
    else if (doGradV)
    {
        const auto output = std::make_tuple(divv, dV11, dV12, dV13, dV22, dV23, dV33);
        neighborhood.ijLoop(input, output, DivVCurlVInteraction<T>{wh}, DivVCurlVPostamble<false, true, T, Tc>{K});
    }
    else
    {
        const auto output = std::make_tuple(divv);
        neighborhood.ijLoop(input, output, DivVCurlVInteraction<T>{wh}, DivVCurlVPostamble<false, false, T, Tc>{K});
    }
}

} // namespace sph
