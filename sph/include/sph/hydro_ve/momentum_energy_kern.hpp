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

#include "sph/kernels.hpp"
#include "sph/math.hpp"
#include "sph/tables.hpp"

namespace sph
{

template<class Tc, class Tm, class T, class Tm1>
HOST_DEVICE_FUN inline void
momentumAndEnergyJLoop(cstone::LocalIndex i, T sincIndex, T K, const cstone::Box<T>& box,
                       const cstone::LocalIndex* neighbors, unsigned neighborsCount, const Tc* x, const Tc* y,
                       const Tc* z, const T* vx, const T* vy, const T* vz, const T* h, const Tm* m, const T* prho,
                       const T* c, const T* c11, const T* c12, const T* c13, const T* c22, const T* c23, const T* c33,
                       const T Atmin, const T Atmax, const T ramp, const T* wh, const T* whd, const T* kx, const T* xm,
                       const T* alpha, const T* dvxdx, const T* dvxdy, const T* dvxdz, const T* dvydx, const T* dvydy,
                       const T* dvydz, const T* dvzdx, const T* dvzdy, const T* dvzdz,
                       T* grad_P_x, T* grad_P_y, T* grad_P_z, Tm1* du, T* maxvsignal)
{
    auto xi  = x[i];
    auto yi  = y[i];
    auto zi  = z[i];
    auto vxi = vx[i];
    auto vyi = vy[i];
    auto vzi = vz[i];

    auto hi  = h[i];
    auto mi  = m[i];
    auto ci  = c[i];
    auto kxi = kx[i];

    auto alpha_i = alpha[i];

    auto xmassi = xm[i];
    auto rhoi   = kxi * mi / xmassi;
    auto prhoi  = prho[i];
    auto voli   = xmassi / kxi;

    T mark_ramp = 0.0;
    T a_mom, b_mom, sigma_ij;

    T hiInv  = T(1) / hi;
    T hiInv3 = hiInv * hiInv * hiInv;

    T maxvsignali = 0.0;
    T momentum_x = 0.0, momentum_y = 0.0, momentum_z = 0.0, energy = 0.0;
    T a_visc_energy = 0.0;

    auto c11i = c11[i];
    auto c12i = c12[i];
    auto c13i = c13[i];
    auto c22i = c22[i];
    auto c23i = c23[i];
    auto c33i = c33[i];

    auto dvxdxi = dvxdx[i];
    auto dvxdyi = dvxdy[i];
    auto dvxdzi = dvxdz[i];
    auto dvydxi = dvydx[i];
    auto dvydyi = dvydy[i];
    auto dvydzi = dvydz[i];
    auto dvzdxi = dvzdx[i];
    auto dvzdyi = dvzdy[i];
    auto dvzdzi = dvzdz[i];

    T eta_crit = std::cbrt(T(32) * M_PI / T(3) / T(neighborsCount + 1)); //+1 is because we need to add selfparticle to neighborsCount

    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[pj];

        T rx     = xi - x[j];
        T ry     = yi - y[j];
        T rz     = zi - z[j];
        auto vxj = vx[j];
        auto vyj = vy[j];
        auto vzj = vz[j];

        applyPBC(box, T(2) * hi, rx, ry, rz);

        T r2   = rx * rx + ry * ry + rz * rz;
        T dist = std::sqrt(r2);

        T vx_ij = vxi - vxj;
        T vy_ij = vyi - vyj;
        T vz_ij = vzi - vzj;

        T hj    = h[j];
        T hjInv = T(1) / hj;

        T v1 = dist * hiInv;
        T v2 = dist * hjInv;


        T hjInv3 = hjInv * hjInv * hjInv;
        T Wi     = hiInv3 * math::pow(lt::wharmonic_lt_with_derivative(wh, whd, v1), (int)sincIndex);
        T Wj     = hjInv3 * math::pow(lt::wharmonic_lt_with_derivative(wh, whd, v2), (int)sincIndex);

        T termA1_i = -(c11i * rx + c12i * ry + c13i * rz) * Wi;
        T termA2_i = -(c12i * rx + c22i * ry + c23i * rz) * Wi;
        T termA3_i = -(c13i * rx + c23i * ry + c33i * rz) * Wi;

        auto c11j = c11[j];
        auto c12j = c12[j];
        auto c13j = c13[j];
        auto c22j = c22[j];
        auto c23j = c23[j];
        auto c33j = c33[j];

        T termA1_j = -(c11j * rx + c12j * ry + c13j * rz) * Wj;
        T termA2_j = -(c12j * rx + c22j * ry + c23j * rz) * Wj;
        T termA3_j = -(c13j * rx + c23j * ry + c33j * rz) * Wj;

        auto mj     = m[j];
        auto cj     = c[j];
        auto kxj    = kx[j];
        auto xmassj = xm[j];
        auto rhoj   = kxj * mj / xmassj;

        T alpha_j   = alpha[j];

        auto dvxdxj = dvxdx[j];
        auto dvxdyj = dvxdy[j];
        auto dvxdzj = dvxdz[j];
        auto dvydxj = dvydx[j];
        auto dvydyj = dvydy[j];
        auto dvydzj = dvydz[j];
        auto dvzdxj = dvzdx[j];
        auto dvzdyj = dvzdy[j];
        auto dvzdzj = dvzdz[j];

        T dmy1 = dvxdxi * rx * rx + dvydxi * rx * ry + dvzdxi * rx * rz +
                 dvxdyi * ry * rx + dvydyi * ry * ry + dvzdyi * ry * rz +
                 dvxdzi * rz * rx + dvydzi * rz * ry + dvzdzi * rz * rz;
        T dmy2 = dvxdxj * rx * rx + dvydxj * rx * ry + dvzdxj * rx * rz +
                 dvxdyj * ry * rx + dvydyj * ry * ry + dvzdyj * ry * rz +
                 dvxdzj * rz * rx + dvydzj * rz * ry + dvzdzj * rz * rz;

        T A_ab = T(0);

        if (dmy2 != T(0)) { A_ab = dmy1 / dmy2;}
        T eta_ab = stl::min(dist / hi, dist / hj);

        T dmy3 = T(1);
        if (eta_ab < eta_crit) {dmy3 = std::exp(- math::pow((eta_ab - eta_crit) / T(0.2), 2));}
        T phi_ab = T(0.5) * stl::max(T(0), stl::min(T(1), T(4) * A_ab / math::pow(T(1) + A_ab, 2))) * dmy3;

        T vAVi_x = vxi - phi_ab * (dvxdxi * rx + dvxdyi * ry + dvxdzi * rz);
        T vAVi_y = vyi - phi_ab * (dvydxi * rx + dvydyi * ry + dvydzi * rz);
        T vAVi_z = vzi - phi_ab * (dvzdxi * rx + dvzdyi * ry + dvzdzi * rz);
        T vAVj_x = vxj + phi_ab * (dvxdxj * rx + dvxdyj * ry + dvxdzj * rz);
        T vAVj_y = vyj + phi_ab * (dvydxj * rx + dvydyj * ry + dvydzj * rz);
        T vAVj_z = vzj + phi_ab * (dvzdxj * rx + dvzdyj * ry + dvzdzj * rz);

        T rv     = rx * (vAVi_x - vAVj_x) + ry * (vAVi_y - vAVj_y) + rz * (vAVi_z - vAVj_z);
        //T rv = rx * vx_ij + ry * vy_ij + rz * vz_ij;

        T wij          = rv / dist;
        T viscosity_ij = artificial_viscosity(alpha_i, alpha_j, ci, cj, wij);
        T viscosity_jj = viscosity_ij;

        // For time-step calculations
        T vijsignal = ci + cj - T(3) * wij;
        maxvsignali = (vijsignal > maxvsignali) ? vijsignal : maxvsignali;

        T prhoj = prho[j];

        T Atwood = (std::abs(rhoi - rhoj)) / (rhoi + rhoj);
        if (Atwood < Atmin)
        {
            a_mom = mj * xmassi * xmassi;
            b_mom = mj * xmassj * xmassj;
        }
        else if (Atwood > Atmax)
        {
            a_mom = mj * xmassi * xmassj;
            b_mom = a_mom;
            mark_ramp += T(1);
        }
        else
        {
            sigma_ij = ramp * (Atwood - Atmin);
            a_mom    = mj * pow(xmassi, T(2) - sigma_ij) * pow(xmassj, sigma_ij);
            b_mom    = mj * pow(xmassj, T(2) - sigma_ij) * pow(xmassi, sigma_ij);
            mark_ramp += sigma_ij;
        }

        auto volj       = xmassj / kxj;
        auto a_visc     = voli * mj / mi * viscosity_ij;
        auto b_visc     = volj * viscosity_jj;
        auto momentum_i = prhoi * a_mom; // + 0.5 * a_visc;
        auto momentum_j = prhoj * b_mom; // + 0.5 * b_visc;
        T    a_visc_x   = T(0.5) * (a_visc * termA1_i + b_visc * termA1_j);
        T    a_visc_y   = T(0.5) * (a_visc * termA2_i + b_visc * termA2_j);
        T    a_visc_z   = T(0.5) * (a_visc * termA3_i + b_visc * termA3_j);

        momentum_x += momentum_i * termA1_i + momentum_j * termA1_j + a_visc_x;
        momentum_y += momentum_i * termA2_i + momentum_j * termA2_j + a_visc_y;
        momentum_z += momentum_i * termA3_i + momentum_j * termA3_j + a_visc_z;

        a_visc_energy += a_visc_x * vx_ij + a_visc_y * vy_ij + a_visc_z * vz_ij;
        energy += momentum_i * (vx_ij * termA1_i + vy_ij * termA2_i + vz_ij * termA3_i);
    }

    a_visc_energy = stl::max(T(0), a_visc_energy);
    du[i]         = K * (energy + T(0.5) * a_visc_energy); // factor of 2 already removed from 2P/rho

    // grad_P_xyz is stored as the acceleration,s accel = -grad_P / rho
    grad_P_x[i] = -K * momentum_x;
    grad_P_y[i] = -K * momentum_y;
    grad_P_z[i] = -K * momentum_z;
    *maxvsignal = maxvsignali;
}

} // namespace sph
