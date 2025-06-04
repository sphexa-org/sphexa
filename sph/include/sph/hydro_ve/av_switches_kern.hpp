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
 * @brief Artifical viscosity switches
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 */

#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/sfc/box.hpp"
#include "cstone/traversal/ijloop/common.hpp"

#include "sph/kernels.hpp"
#include "sph/table_lookup.hpp"

namespace sph
{

template<class T, class Tc>
struct AVswitchesInteraction
{
    const T* wh;
    Tc       K;

    template<class ParticleData>
    constexpr auto operator()(const ParticleData& iData, const ParticleData& jData, cstone::Vec3<Tc> const& r_ij,
                              T r2) const
    {
        const auto [i, iPos, hi, xmi, kxi, divv_i, alpha_i, vxi, vyi, vzi, ci, c11i, c12i, c13i, c22i, c23i, c33i] =
            iData;
        const auto [j, jPos, hj, xmj, kxj, divv_j, alpha_j, vxj, vyj, vzj, cj, c11j, c12j, c13j, c22j, c23j, c33j] =
            jData;

        T rx = r_ij[0];
        T ry = r_ij[1];
        T rz = r_ij[2];

        auto hiInv  = T(1) / hi;
        auto hiInv3 = hiInv * hiInv * hiInv;

        T dist = std::sqrt(r2);

        T vx_ij = vxi - vxj;
        T vy_ij = vyi - vyj;
        T vz_ij = vzi - vzj;

        T rv           = rx * vx_ij + ry * vy_ij + rz * vz_ij;
        T vijsignal_ij = 0.0;

        if (rv < T(0)) { vijsignal_ij = ci + cj - T(3) * rv / dist; }

        T v1 = dist * hiInv;
        T Wi = K * hiInv3 * lt::lookup(wh, v1);

        T termA1 = -(c11i * rx + c12i * ry + c13i * rz) * Wi;
        T termA2 = -(c12i * rx + c22i * ry + c23i * rz) * Wi;
        T termA3 = -(c13i * rx + c23i * ry + c33i * rz) * Wi;

        T volj   = xmj / kxj;
        T factor = volj * (divv_i - divv_j);

        T graddivv_x = factor * termA1;
        T graddivv_y = factor * termA2;
        T graddivv_z = factor * termA3;
        return std::make_tuple(graddivv_x, graddivv_y, graddivv_z, cstone::ijloop::reduction::max(vijsignal_ij));
    }
};

template<class T, class Tc>
struct AVswitchesPostamble
{
    T  alphamin, alphamax, decay_constant;
    Tc dt;

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        auto [i, iPos, hi, xmi, kxi, divv_i, alpha_i, vxi, vyi, vzi, ci, c11i, c12i, c13i, c22i, c23i, c33i] = iData;
        auto [graddivv_x, graddivv_y, graddivv_z, vijsignalr_i]                                              = result;
        T vijsignal_i = std::max(vijsignalr_i, T(1e-40));

        T graddivv = std::sqrt(graddivv_x * graddivv_x + graddivv_y * graddivv_y + graddivv_z * graddivv_z);

        T alphaloc = 0;
        if (divv_i < T(0))
        {
            T a_const = hi * hi * graddivv;
            alphaloc  = alphamax * a_const / (a_const + hi * std::abs(divv_i) + T(0.05) * ci);
        }

        if (alphaloc >= alpha_i) { alpha_i = alphaloc; }
        else
        {
            T decay    = hi / (decay_constant * vijsignal_i);
            T alphadot = 0.0;
            if (alphaloc >= alphamin) { alphadot = (alphaloc - alpha_i) / decay; }
            else { alphadot = (alphamin - alpha_i) / decay; }
            alpha_i += alphadot * dt;
        }

        return std::make_tuple(alpha_i);
    }
};

template<size_t stride = 1, class Tc, class T>
HOST_DEVICE_FUN inline T
AVswitchesJLoop(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box, const cstone::LocalIndex* neighbors,
                unsigned neighborsCount, const Tc* x, const Tc* y, const Tc* z, const T* vx, const T* vy, const T* vz,
                const T* h, const T* c, const T* c11, const T* c12, const T* c13, const T* c22, const T* c23,
                const T* c33, const T* wh, const T* /*whd*/, const T* kx, const T* xm, const T* divv, const T* alpha,
                const Tc dt, const T alphamin, const T alphamax, const T decay_constant)
{
    AVswitchesInteraction<T, Tc> interaction{wh, K};
    AVswitchesPostamble<T, Tc>   postamble{alphamin, alphamax, decay_constant, dt};

    const auto input   = std::make_tuple(xm, kx, divv, alpha, vx, vy, vz, c, c11, c12, c13, c22, c23, c33);
    T          alpha_i = 0;
    const auto output  = std::make_tuple((&alpha_i) - i);

    const auto iData  = cstone::ijloop::loadParticleData(x, y, z, h, input, i);
    const bool usePbc = cstone::ijloop::requiresPbcHandling(box, iData);

    auto result = interaction(iData, iData, cstone::Vec3<Tc>{0, 0, 0}, T(0));
    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[stride * pj];

        const auto jData = cstone::ijloop::loadParticleData(x, y, z, h, input, j);

        const auto [r_ij, r2] = cstone::ijloop::posDiffAndDistSq(usePbc, box, iData, jData);

        cstone::ijloop::updateResult(result, interaction(iData, jData, r_ij, r2));
    }

    auto presult = postamble(iData, cstone::ijloop::unwrapModifiers(result));

    cstone::ijloop::storeParticleData(output, i, presult);

    return alpha_i;
}

template<class Neighborhood, class Tc, class T>
void AVswitchesIjLoop(Neighborhood const& neighborhood, Tc K, Tc dt, T alphamin, T alphamax, T decay_constant,
                      const T* xm, const T* kx, const T* divv, const T* alpha, const T* vx, const T* vy, const T* vz,
                      const T* c, const T* c11, const T* c12, const T* c13, const T* c22, const T* c23, const T* c33,
                      const T* wh, T* alpha_out)
{
    neighborhood.ijLoop(std::make_tuple(xm, kx, divv, alpha, vx, vy, vz, c, c11, c12, c13, c22, c23, c33),
                        std::make_tuple(alpha_out), AVswitchesInteraction<T, Tc>{wh, K},
                        AVswitchesPostamble<T, Tc>{alphamin, alphamax, decay_constant, dt});
}

} // namespace sph
