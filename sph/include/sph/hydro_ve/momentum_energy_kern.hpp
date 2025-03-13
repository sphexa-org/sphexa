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
 * @brief Pressure gradients and energy kernel
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 */

#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/sfc/box.hpp"
#include "cstone/traversal/ijloop/ijloop.hpp"

#include "sph/kernels.hpp"
#include "sph/table_lookup.hpp"

namespace sph
{

template<class Tc, class T>
HOST_DEVICE_FUN T avRvCorrection(util::array<Tc, 3> R, Tc eta_ab, T eta_crit, const util::array<T, 6>& gradV_i,
                                 const util::array<const T, 6>& gradV_j)
{
    T dmy1 = dot(R, symv(gradV_i, R));
    T dmy2 = dot(R, symv(gradV_j, R));
    T dmy3 = T(1);
    if (eta_ab < eta_crit)
    {
        T etaDiff = T(5) * (eta_ab - eta_crit);
        dmy3      = std::exp(-etaDiff * etaDiff);
    }

    T A_ab   = (dmy2 != T(0)) ? dmy1 / dmy2 : T(0);
    T A_abp1 = T(1) + A_ab;
    T phi_ab = T(0.5) * dmy3 * stl::max(T(0), stl::min(T(1), T(4) * A_ab / (A_abp1 * A_abp1)));

    // additive AV correction to rv = dot(R, Vij)
    T rv_AV = -phi_ab * (dmy1 + dmy2);
    return rv_AV;
}

template<bool AvClean, class T>
struct MomentumAndEnergyInteraction
{
    const T* wh;
    T        Atmin, Atmax, ramp;

    template<class ParticleData, class Tc>
    constexpr auto operator()(const ParticleData& iData, const ParticleData& jData, cstone::Vec3<Tc> const& r_ij,
                              T r2) const
    {
        const auto [i, iPos, hi, vxi, vyi, vzi, mi, ci, kxi, alpha_i, xmassi, prhoi, c11i, c12i, c13i, c22i, c23i, c33i,
                    nci, dV11i, dV12i, dV13i, dV22i, dV23i, dV33i, tdpdTrhoi] = iData;
        const auto [j, jPos, hj, vxj, vyj, vzj, mj, cj, kxj, alpha_j, xmassj, prhoj, c11j, c12j, c13j, c22j, c23j, c33j,
                    ncj, dV11j, dV12j, dV13j, dV22j, dV23j, dV33j, tdpdTrhoj] = jData;

        auto rhoi = kxi * mi / xmassi;

        T hiInv  = T(1) / hi;
        T hiInv3 = hiInv * hiInv * hiInv;

        T eta_crit = std::cbrt(T(32) * M_PI / T(3) / T(nci));

        T rx = r_ij[0];
        T ry = r_ij[1];
        T rz = r_ij[2];

        T dist = std::sqrt(r2);

        T vx_ij = vxi - vxj;
        T vy_ij = vyi - vyj;
        T vz_ij = vzi - vzj;

        T hjInv = T(1) / hj;

        T v1 = dist * hiInv;
        T v2 = dist * hjInv;

        T hjInv3 = hjInv * hjInv * hjInv;
        T Wi     = hiInv3 * lt::lookup(wh, v1);
        T Wj     = hjInv3 * lt::lookup(wh, v2);

        T termA1_i = -(c11i * rx + c12i * ry + c13i * rz) * Wi;
        T termA2_i = -(c12i * rx + c22i * ry + c23i * rz) * Wi;
        T termA3_i = -(c13i * rx + c23i * ry + c33i * rz) * Wi;

        T termA1_j = -(c11j * rx + c12j * ry + c13j * rz) * Wj;
        T termA2_j = -(c12j * rx + c22j * ry + c23j * rz) * Wj;
        T termA3_j = -(c13j * rx + c23j * ry + c33j * rz) * Wj;

        auto rhoj = kxj * mj / xmassj;

        T rv = rx * vx_ij + ry * vy_ij + rz * vz_ij;
        if constexpr (AvClean)
        {
            rv += avRvCorrection({rx, ry, rz}, stl::min(v1, v2), eta_crit, {dV11i, dV12i, dV13i, dV22i, dV23i, dV33i},
                                 {dV11j, dV12j, dV13j, dV22j, dV23j, dV33j});
        }

        T wij          = i == j ? 0 : rv / dist;
        T viscosity_ij = artificial_viscosity(alpha_i, alpha_j, ci, cj, wij);

        // For time-step calculations
        T vijsignal = i == j ? 0 : T(0.5) * (ci + cj) - T(2) * wij;

        T a_mom, b_mom;
        T Atwood = (std::abs(rhoi - rhoj)) / (rhoi + rhoj);
        if (Atwood < Atmin)
        {
            a_mom = xmassi * xmassi;
            b_mom = xmassj * xmassj;
        }
        else if (Atwood > Atmax)
        {
            a_mom = xmassi * xmassj;
            b_mom = a_mom;
        }
        else
        {
            T sigma_ij = ramp * (Atwood - Atmin);
            a_mom      = pow(xmassi, T(2) - sigma_ij) * pow(xmassj, sigma_ij);
            b_mom      = pow(xmassj, T(2) - sigma_ij) * pow(xmassi, sigma_ij);
        }

        auto a_visc        = mj / rhoi * viscosity_ij;
        auto b_visc        = mj / rhoj * viscosity_ij;
        T    a_visc_x      = T(0.5) * (a_visc * termA1_i + b_visc * termA1_j);
        T    a_visc_y      = T(0.5) * (a_visc * termA2_i + b_visc * termA2_j);
        T    a_visc_z      = T(0.5) * (a_visc * termA3_i + b_visc * termA3_j);
        T    a_visc_energy = a_visc_x * vx_ij + a_visc_y * vy_ij + a_visc_z * vz_ij;

        T energy = mj * a_mom * (vx_ij * termA1_i + vy_ij * termA2_i + vz_ij * termA3_i);

        auto momentum_i = mj * prhoi * a_mom;
        auto momentum_j = mj * prhoj * b_mom;
        T    momentum_x = momentum_i * termA1_i + momentum_j * termA1_j + a_visc_x;
        T    momentum_y = momentum_i * termA2_i + momentum_j * termA2_j + a_visc_y;
        T    momentum_z = momentum_i * termA3_i + momentum_j * termA3_j + a_visc_z;

        return std::make_tuple(a_visc_energy, energy, momentum_x, momentum_y, momentum_z,
                               cstone::ijloop::reduction::max(vijsignal));
    }
};

template<bool UseTdpdTrho, class T, class Tc>
struct MomentumAndEnergyPostamble
{
    Tc K;

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        const auto [i, iPos, hi, vxi, vyi, vzi, mi, ci, kxi, alpha_i, xmassi, prhoi, c11i, c12i, c13i, c22i, c23i, c33i,
                    nci, dV11i, dV12i, dV13i, dV22i, dV23i, dV33i, tdpdTrhoi]        = iData;
        auto [a_visc_energy, energy, momentum_x, momentum_y, momentum_z, maxvsignal] = result;
        a_visc_energy                                                                = stl::max(T(0), a_visc_energy);
        T eCoeff                                                                     = UseTdpdTrho ? tdpdTrhoi : prhoi;
        T dui = K * (eCoeff * energy + T(0.5) * a_visc_energy); // factor of 2 already removed from 2P/rho

        // grad_P_xyz is stored as the acceleration,s accel = -grad_P / rho
        return std::make_tuple(dui, -K * momentum_x, -K * momentum_y, -K * momentum_z, maxvsignal);
    };
};

template<bool UseTdpdTrho, class T, class Tc>
struct MomentumAndEnergyPostambleWithDt : MomentumAndEnergyPostamble<UseTdpdTrho, T, Tc>
{
    Tc Kcour;

    MomentumAndEnergyPostambleWithDt(Tc K, Tc Kcour)
        : MomentumAndEnergyPostamble<UseTdpdTrho, T, Tc>{K}
        , Kcour(Kcour)
    {
    }

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        const auto [du, grad_P_x, grad_P_y, grad_P_z, maxvsignal] =
            MomentumAndEnergyPostamble<UseTdpdTrho, T, Tc>::operator()(iData, result);
        const auto [i, iPos, hi, vxi, vyi, vzi, mi, ci, kxi, alpha_i, xmassi, prhoi, c11i, c12i, c13i, c22i, c23i, c33i,
                    nci, dV11i, dV12i, dV13i, dV22i, dV23i, dV33i, tdpdTrhoi] = iData;

        auto dt = tsKCourant(maxvsignal.value, hi, ci, Kcour);
        return std::make_tuple(du, grad_P_x, grad_P_y, grad_P_z, dt);
    };
};

template<bool avClean, size_t stride = 1, class Tc, class Tm, class T, class Tm1>
HOST_DEVICE_FUN inline void
momentumAndEnergyJLoop(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box, const cstone::LocalIndex* neighbors,
                       unsigned neighborsCount, const unsigned* nc, const Tc* x, const Tc* y, const Tc* z, const T* vx,
                       const T* vy, const T* vz, const T* h, const Tm* m, const T* prho, const T* tdpdTrho, const T* c,
                       const T* c11, const T* c12, const T* c13, const T* c22, const T* c23, const T* c33,
                       const T Atmin, const T Atmax, const T ramp, const T* wh, const T* kx, const T* xm,
                       const T* alpha, const T* dV11, const T* dV12, const T* dV13, const T* dV22, const T* dV23,
                       const T* dV33, T* grad_P_x, T* grad_P_y, T* grad_P_z, Tm1* du, T* maxvsignal)
{
    MomentumAndEnergyInteraction<avClean, T> interaction{wh, Atmin, Atmax, ramp};

    if constexpr (!avClean) dV11 = dV12 = dV13 = dV22 = dV23 = dV33 = vx;
    const auto input =
        std::make_tuple(vx, vy, vz, m, c, kx, alpha, xm, prho, c11, c12, c13, c22, c23, c33, nc, dV11, dV12, dV13, dV22,
                        dV23, dV33, tdpdTrho ? tdpdTrho : vx /* pass random derefable array if tdpdTrho is null */);
    const auto output = std::make_tuple(du, grad_P_x, grad_P_y, grad_P_z, maxvsignal - i);

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

    if (tdpdTrho)
    {
        MomentumAndEnergyPostamble<true, T, Tc> postamble{K};
        auto                                    presult = postamble(iData, result);
        cstone::ijloop::storeParticleData(output, i, presult);
    }
    else
    {
        MomentumAndEnergyPostamble<false, T, Tc> postamble{K};
        auto                                     presult = postamble(iData, result);
        cstone::ijloop::storeParticleData(output, i, presult);
    }
}

template<bool AvClean, class Neighborhood, class Tc, class T, class Tm, class Tm1>
void momentumAndEnergyIjLoop(Neighborhood const& neighborhood, Tc K, Tc Kcour, T Atmin, T Atmax, T ramp, const T* vx,
                             const T* vy, const T* vz, const Tm* m, const T* c, const T* kx, const T* alpha,
                             const T* xm, const T* prho, const T* c11, const T* c12, const T* c13, const T* c22,
                             const T* c23, const T* c33, const unsigned* nc, const T* dV11, const T* dV12,
                             const T* dV13, const T* dV22, const T* dV23, const T* dV33, const T* tdpdTrho, const T* wh,
                             Tm1* du, T* grad_P_x, T* grad_P_y, T* grad_P_z, T* dt)
{
    if constexpr (!AvClean) dV11 = dV12 = dV13 = dV22 = dV23 = dV33 = vx;
    const auto input =
        std::make_tuple(vx, vy, vz, m, c, kx, alpha, xm, prho, c11, c12, c13, c22, c23, c33, nc, dV11, dV12, dV13, dV22,
                        dV23, dV33, tdpdTrho ? tdpdTrho : vx /* pass random derefable array if tdpdTrho is null */);
    const auto output = std::make_tuple(du, grad_P_x, grad_P_y, grad_P_z, dt);
    if (tdpdTrho)
    {
        neighborhood.ijLoop(input, output, MomentumAndEnergyInteraction<AvClean, T>{wh, Atmin, Atmax, ramp},
                            MomentumAndEnergyPostambleWithDt<true, T, Tc>{K, Kcour},
                            cstone::ijloop::symmetry::asymmetric);
    }
    else
    {
        neighborhood.ijLoop(input, output, MomentumAndEnergyInteraction<AvClean, T>{wh, Atmin, Atmax, ramp},
                            MomentumAndEnergyPostambleWithDt<false, T, Tc>{K, Kcour},
                            cstone::ijloop::symmetry::asymmetric);
    }
}

} // namespace sph
