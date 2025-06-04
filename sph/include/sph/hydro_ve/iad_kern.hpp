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
#include "cstone/traversal/ijloop/common.hpp"

#include "sph/kernels.hpp"
#include "sph/table_lookup.hpp"

namespace sph
{

template<class T>
struct IADInteraction
{
    const T* wh;

    template<class ParticleData, class Tc>
    constexpr auto operator()(const ParticleData& iData, const ParticleData& jData, cstone::Vec3<Tc> const& r_ij,
                              T r2) const
    {
        const auto [i, iPos, hi, xmi, kxi] = iData;
        const auto [j, jPos, hj, xmj, kxj] = jData;

        T rx = r_ij[0];
        T ry = r_ij[1];
        T rz = r_ij[2];

        T hiInv = T(1) / hi;

        T dist = std::sqrt(r2);

        T vloc = dist * hiInv;
        T w    = lt::lookup(wh, vloc);

        T volj_w = xmj / kxj * w;

        return std::make_tuple(rx * rx * volj_w, //
                               rx * ry * volj_w, //
                               rx * rz * volj_w, //
                               ry * ry * volj_w, //
                               ry * rz * volj_w, //
                               rz * rz * volj_w);
    }
};

template<class T, class Tc>
struct IADPostamble
{
    Tc K;

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        const auto [i, iPos, hi, mi, roi]               = iData;
        auto [tau11, tau12, tau13, tau22, tau23, tau33] = result;

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

        // Note normalization factor: cij have units of 1/tau because det is proportional to tau^3 so we have to
        // divide by K/h^3.
        T factor = normalization * (hi * hi * hi) / (det * K);

        return std::make_tuple(                       //
            (tau22 * tau33 - tau23 * tau23) * factor, //
            (tau13 * tau23 - tau33 * tau12) * factor, //
            (tau12 * tau23 - tau22 * tau13) * factor, //
            (tau11 * tau33 - tau13 * tau13) * factor, //
            (tau13 * tau12 - tau11 * tau23) * factor, //
            (tau11 * tau22 - tau12 * tau12) * factor);
    }
};

template<size_t stride = 1, class Tc, class T>
HOST_DEVICE_FUN inline void IADJLoop(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box,
                                     const cstone::LocalIndex* neighbors, unsigned neighborsCount, const Tc* x,
                                     const Tc* y, const Tc* z, const T* h, const T* wh, const T* /*whd*/, const T* xm,
                                     const T* kx, T* c11, T* c12, T* c13, T* c22, T* c23, T* c33)
{
    IADInteraction      interaction{wh};
    IADPostamble<T, Tc> postamble{K};

    const auto input  = std::make_tuple(xm, kx);
    const auto output = std::make_tuple(c11, c12, c13, c22, c23, c33);

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
}

template<class Neighborhood, class Tc, class T>
void IADIjLoop(Neighborhood const& neighborhood, Tc K, const T* xm, const T* kx, const T* wh, T* c11, T* c12, T* c13,
               T* c22, T* c23, T* c33)
{
    neighborhood.ijLoop(std::make_tuple(xm, kx), std::make_tuple(c11, c12, c13, c22, c23, c33), IADInteraction<T>{wh},
                        IADPostamble<T, Tc>{K});
}

} // namespace sph
