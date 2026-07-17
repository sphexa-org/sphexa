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

#include <cstdint>

#include "cstone/traversal/ijloop/ijloop.hpp"

#include "sph/iad_regularization.hpp"
#include "sph/table_lookup.hpp"

namespace sph
{

template<class T>
struct IADGradhInteraction
{
    const T *wh, *whd;

    template<class ParticleData, class Tc>
    constexpr auto operator()(const ParticleData& iData, const ParticleData& jData, cstone::Vec3<Tc> const& r_ij,
                              T r2) const
    {
        const auto [i, iPos, hi, mi, xmi, kxi, nci, idi] = iData;
        const auto [j, jPos, hj, mj, xmj, kxj, ncj, idj] = jData;

        T rx = r_ij[0];
        T ry = r_ij[1];
        T rz = r_ij[2];

        T hiInv = T(1) / hi;

        T dist = std::sqrt(r2);
        T vloc = dist * hiInv;
        T w    = i == j ? 0 : lt::lookup(wh, vloc);

        T volj_w = xmj / kxj * w;

        T tau11 = rx * rx * volj_w;
        T tau12 = rx * ry * volj_w;
        T tau13 = rx * rz * volj_w;
        T tau22 = ry * ry * volj_w;
        T tau23 = ry * rz * volj_w;
        T tau33 = rz * rz * volj_w;

        // gradh summation
        T dw        = i == j ? 0 : lt::lookup(whd, vloc);
        T dterh     = -(T(3) * w + vloc * dw);
        T whomegai  = dterh * xmj;
        T wrho0i    = dterh * mj;
        T sum_error = dterh * xmj / kxj;

        return std::make_tuple(tau11, tau12, tau13, tau22, tau23, tau33, whomegai, wrho0i, sum_error);
    }
};

template<class T, class Tc>
struct IADGradhPostamble
{
    const Tc       K;
    const T        iadConditionQuality{};
    const unsigned iadRegBit;

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        const auto [i, iPos, hi, mi, xmi, kxi, nci, idi]                            = iData;
        auto [tau11, tau12, tau13, tau22, tau23, tau33, whomegai, wrho0i, sum_error] = result;

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

        auto [det, regularize] = needRegularization(tau11, tau12, tau13, tau22, tau23, tau33, iadConditionQuality);
        if (regularize) { regularizeIadMomentMatrix(tau11, tau12, tau13, tau22, tau23, tau33, iadConditionQuality); }
        uint64_t newId = setRegularizationTag(regularize, iadRegBit, idi);

        // note normalization factor: cij have units of 1/tau because det is proportional to tau^3, so we have to
        // divide by K/h^3
        T factor = (nci > 1 && det > T(0)) ? normalization * (hi * hi * hi) / (det * K) : T(0);
        if (not std::isfinite(factor)) { factor = T(0); }

        T c11i = (tau22 * tau33 - tau23 * tau23) * factor;
        T c12i = (tau13 * tau23 - tau33 * tau12) * factor;
        T c13i = (tau12 * tau23 - tau22 * tau13) * factor;
        T c22i = (tau11 * tau33 - tau13 * tau13) * factor;
        T c23i = (tau13 * tau12 - tau11 * tau23) * factor;
        T c33i = (tau11 * tau22 - tau12 * tau12) * factor;
        // -----------------------------------------------------

        // -----------------------------------------------------
        // gradh postamble
        T    hiInv = T(1) / hi;
        auto h3Inv = hiInv * hiInv * hiInv;
        auto dnorm = K * hiInv * h3Inv;

        whomegai *= dnorm;
        wrho0i *= dnorm;
        sum_error *= dnorm;

        T rhoi = kxi * mi / xmi;

        // The following line uses kxi instead of rhoi/rho0i so that it doesn't need to save an extra variable.
        // It is correct. However, assumes that the VE definition is xmass=mass/rho.
        // If the VE definition changes, this line needs to be updated accordingly.
        whomegai = whomegai * mi / xmi - rhoi * sum_error + (kxi - K * xmi * h3Inv) * (wrho0i - rhoi / kxi * sum_error);
        T dhdrho = -hi / (rhoi * T(3)); // This /3 is the dimension hard-coded.

        T gradhi = T(1) - dhdrho * whomegai;
        return std::make_tuple(c11i, c12i, c13i, c22i, c23i, c33i, gradhi, newId);
    }
};

} // namespace sph
