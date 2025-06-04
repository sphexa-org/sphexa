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
#include "cstone/traversal/ijloop/common.hpp"

#include "sph/kernels.hpp"
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

        cstone::Vec3<T> termA;
        termA[0] = -(c11i * rx + c12i * ry + c13i * rz) * Wi;
        termA[1] = -(c12i * rx + c22i * ry + c23i * rz) * Wi;
        termA[2] = -(c13i * rx + c23i * ry + c33i * rz) * Wi;

        return std::make_tuple((vx_ji * xmj) * termA, (vy_ji * xmj) * termA, (vz_ji * xmj) * termA);
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
        const auto [dVxi, dVyi, dVzi]                                                         = result;

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

template<size_t stride = 1, typename Tc, class T>
HOST_DEVICE_FUN inline void
divV_curlVJLoop(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box, const cstone::LocalIndex* neighbors,
                unsigned neighborsCount, const Tc* x, const Tc* y, const Tc* z, const T* vx, const T* vy, const T* vz,
                const T* h, const T* c11, const T* c12, const T* c13, const T* c22, const T* c23, const T* c33,
                const T* wh, const T* /*whd*/, const T* kx, const T* xm, T* divv, T* curlv, T* dV11, T* dV12, T* dV13,
                T* dV22, T* dV23, T* dV33, bool doGradV)
{
    DivVCurlVInteraction<T> interaction{wh};

    const auto input = std::make_tuple(vx, vy, vz, xm, kx, c11, c12, c13, c22, c23, c33);

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

    if (curlv && doGradV)
    {
        DivVCurlVPostamble<true, true, T, Tc> postamble{K};
        const auto                            presult = postamble(iData, cstone::ijloop::unwrapModifiers(result));
        const auto                            output = std::make_tuple(divv, curlv, dV11, dV12, dV13, dV22, dV23, dV33);
        cstone::ijloop::storeParticleData(output, i, presult);
    }
    else if (curlv)
    {
        DivVCurlVPostamble<true, false, T, Tc> postamble{K};
        const auto                             presult = postamble(iData, cstone::ijloop::unwrapModifiers(result));
        const auto                             output  = std::make_tuple(divv, curlv);
        cstone::ijloop::storeParticleData(output, i, presult);
    }
    else if (doGradV)
    {
        DivVCurlVPostamble<false, true, T, Tc> postamble{K};
        const auto                             presult = postamble(iData, cstone::ijloop::unwrapModifiers(result));
        const auto                             output  = std::make_tuple(divv, dV11, dV12, dV13, dV22, dV23, dV33);
        cstone::ijloop::storeParticleData(output, i, presult);
    }
    else
    {
        DivVCurlVPostamble<false, false, T, Tc> postamble{K};
        const auto                              presult = postamble(iData, cstone::ijloop::unwrapModifiers(result));
        const auto                              output  = std::make_tuple(divv);
        cstone::ijloop::storeParticleData(output, i, presult);
    }
}

template<class Neighborhood, class Tc, class T>
void divVCurlVIjLoop(const Neighborhood& neighborhood, Tc K, const T* vx, const T* vy, const T* vz, const T* xm,
                     const T* kx, const T* c11, const T* c12, const T* c13, const T* c22, const T* c23, const T* c33,
                     const T* wh, T* divv, T* curlv, T* dV11, T* dV12, T* dV13, T* dV22, T* dV23, T* dV33, bool doGradV)
{
    const auto input = std::make_tuple(vx, vy, vz, xm, kx, c11, c12, c13, c22, c23, c33);
    // TODO: check symmetry
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
