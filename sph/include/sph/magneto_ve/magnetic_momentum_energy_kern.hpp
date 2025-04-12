/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUTh WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUTh NOTh LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENTh SHALL THE
 * AUTHORS OR COPYRIGHTh HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORTh OR OTHERWISE, ARISING FROM,
 * OUTh OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file calculates the momentum and energy with the magnetic stress tensor
 *
 * @author Lukas Schmidt
 *
 */

#include "cstone/cuda/annotation.hpp"
#include "cstone/sfc/box.hpp"
#include "sph/hydro_ve/momentum_energy_kern.hpp"

#include "sph/kernels.hpp"
#include "sph/table_lookup.hpp"

namespace sph::magneto
{
/*! @brief calculates the momentum contributions with the magnetic stress tensor
 *
 */
template<bool avClean, size_t stride = 1, class Tc, class Tm, class T, class Tm1>
HOST_DEVICE_FUN inline void magneticMomentumJLoop(
    cstone::LocalIndex i, Tc K, const Tc mu_0, const cstone::Box<Tc>& box, const cstone::LocalIndex* neighbors,
    unsigned neighborsCount, const Tc* x, const Tc* y, const Tc* z, const T* vx, const T* vy, const T* vz, const T* h,
    const Tm* m, const T* p, const T* tdpdTrho, const T* c, const T* c11, const T* c12, const T* c13, const T* c22,
    const T* c23, const T* c33, const T Atmin, const T Atmax, const T ramp, const T* wh, const T* kx, const T* xm,
    const T* alpha, const T* dvxdx, const T* dvxdy, const T* dvxdz, const T* dvydx, const T* dvydy, const T* dvydz,
    const T* dvzdx, const T* dvzdy, const T* dvzdz, const Tc* Bx, const Tc* By, const Tc* Bz, const T* gradh,
    T* grad_P_x, T* grad_P_y, T* grad_P_z, Tm1* du, T* maxvsignal)
{

    T    mu_0Inv = 1 / mu_0;
    auto Bxi     = Bx[i];
    auto Byi     = By[i];
    auto Bzi     = Bz[i];
    auto pi      = p[i];

    T Si_xx = 0.5 * mu_0Inv * (Bxi * Bxi - Byi * Byi - Bzi * Bzi);
    T Si_xy = mu_0Inv * Bxi * Byi;
    T Si_xz = mu_0Inv * Bxi * Bzi;
    T Si_yy = 0.5 * mu_0Inv * (-Bxi * Bxi + Byi * Byi - Bzi * Bzi);
    T Si_yz = mu_0Inv * Byi * Bzi;
    T Si_zz = 0.5 * mu_0Inv * (-Bxi * Bxi - Byi * Byi + Bzi * Bzi);

    auto xi  = x[i];
    auto yi  = y[i];
    auto zi  = z[i];
    auto vxi = vx[i];
    auto vyi = vy[i];
    auto vzi = vz[i];

    auto hi     = h[i];
    auto mi     = m[i];
    auto ci     = c[i];
    auto kxi    = kx[i];
    auto gradhi = gradh[i];

    auto alpha_i = alpha[i];

    auto xmassi = xm[i];
    auto rhoi   = kxi * mi / xmassi;

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

    [[maybe_unused]] util::array<T, 6> gradV_i;
    if constexpr (avClean)
    {
        gradV_i = {dvxdx[i], dvxdy[i] + dvydx[i], dvxdz[i] + dvzdx[i], dvydy[i], dvydz[i] + dvzdy[i], dvzdz[i]};
    }

    // +1 is because we need to add selfparticle to neighborsCount
    T eta_crit = std::cbrt(T(32) * M_PI / T(3) / T(neighborsCount + 1));

    // for tensile instability correction
    T norm2_B = Bxi * Bxi + Byi * Byi + Bzi * Bzi;
    T beta    = 2 * mu_0 * pi / norm2_B;
    T H       = 0.;
    // if (beta < 1) { H = 2.; } //SPHYNX
    // else if (beta <= 2) { H = 2 * (2. - beta); }
    if (beta < 2.) { H = 1.; } // PHANTOM
    else if (beta < 10.) { H = (10. - beta) / 8.; }
    // if (beta < 1.) { H = 1.; } //Wissing et al.
    // else if (beta < 2.) { H = 2. - beta; }

    T f_i = 0.0;

    T v_alfven2i       = norm2_B / (mu_0 * rhoi);
    T magneticVsignali = std::sqrt(ci * ci + v_alfven2i);

    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[stride * pj];

        T    rx  = xi - x[j];
        T    ry  = yi - y[j];
        T    rz  = zi - z[j];
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
        T Wi     = hiInv3 * lt::lookup(wh, v1);
        T Wj     = hjInv3 * lt::lookup(wh, v2);

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

        T rv = rx * vx_ij + ry * vy_ij + rz * vz_ij;
        if constexpr (avClean)
        {
            rv += avRvCorrection(
                {rx, ry, rz}, stl::min(v1, v2), eta_crit, gradV_i,
                {dvxdx[j], dvxdy[j] + dvydx[j], dvxdz[j] + dvzdx[j], dvydy[j], dvydz[j] + dvzdy[j], dvzdz[j]});
        }

        T v_alfven2j       = (Bx[j] * Bx[j] + By[j] * By[j] + Bz[j] * Bz[j]) / (rhoj * mu_0);
        T magneticVsignalj = std::sqrt(cj * cj + v_alfven2j);

        T wij          = rv / dist;
        T viscosity_ij = artificial_viscosity(alpha_i, alpha[j], magneticVsignali, magneticVsignalj, wij);

        // For time-step calculations
        T vijsignal = magneticVsignali + magneticVsignalj - T(3) * wij;
        maxvsignali = (vijsignal > maxvsignali) ? vijsignal : maxvsignali;

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

        auto a_visc   = mj / (rhoi * gradhi) * viscosity_ij;
        auto b_visc   = mj / (rhoj * gradh[j]) * viscosity_ij;
        T    a_visc_x = T(0.5) * (a_visc * termA1_i + b_visc * termA1_j);
        T    a_visc_y = T(0.5) * (a_visc * termA2_i + b_visc * termA2_j);
        T    a_visc_z = T(0.5) * (a_visc * termA3_i + b_visc * termA3_j);
        a_visc_energy += a_visc_x * vx_ij + a_visc_y * vy_ij + a_visc_z * vz_ij;

        energy += mj * a_mom * (vx_ij * termA1_i + vy_ij * termA2_i + vz_ij * termA3_i);

        T Sj_xx = 0.5 * mu_0Inv * (Bx[j] * Bx[j] - By[j] * By[j] - Bz[j] * Bz[j]);
        T Sj_xy = mu_0Inv * Bx[j] * By[j];
        T Sj_xz = mu_0Inv * Bx[j] * Bz[j];
        T Sj_yy = 0.5 * mu_0Inv * (-Bx[j] * Bx[j] + By[j] * By[j] - Bz[j] * Bz[j]);
        T Sj_yz = mu_0Inv * By[j] * Bz[j];
        T Sj_zz = 0.5 * mu_0Inv * (-Bx[j] * Bx[j] - By[j] * By[j] + Bz[j] * Bz[j]);

        // gas pressure contributions
        a_mom /= kxi * mi * mi * gradhi;
        b_mom /= kxj * mj * mj * gradh[j];

        auto momentum_i = mj * pi * a_mom;
        auto momentum_j = mj * p[j] * b_mom;
        momentum_x -= momentum_i * termA1_i + momentum_j * termA1_j;
        momentum_y -= momentum_i * termA2_i + momentum_j * termA2_j;
        momentum_z -= momentum_i * termA3_i + momentum_j * termA3_j;

        // magnetic pressure contributions
        auto momentum_xi = Si_xx * termA1_i + Si_xy * termA2_i + Si_xz * termA3_i;
        auto momentum_yi = Si_xy * termA1_i + Si_yy * termA2_i + Si_yz * termA3_i;
        auto momentum_zi = Si_xz * termA1_i + Si_yz * termA2_i + Si_zz * termA3_i;

        auto momentum_xj = Sj_xx * termA1_j + Sj_xy * termA2_j + Sj_xz * termA3_j;
        auto momentum_yj = Sj_xy * termA1_j + Sj_yy * termA2_j + Sj_yz * termA3_j;
        auto momentum_zj = Sj_xz * termA1_j + Sj_yz * termA2_j + Sj_zz * termA3_j;

        auto rhosqinv_i = 1 / (rhoi * rhoi * gradhi);
        auto rhosqinv_j = 1 / (rhoj * rhoj * gradh[j]);

        // tensile instability correction
        // auto rhosqinv = 1 / (rhoi * rhoj * gradhi);
        // f_i += 2 * mj * rhosqinv * (Bxi * termA1_i + Byi * termA2_i + Bzi * termA3_i); // SPHYNX
        // f_i += mj / (rhoi * rhoj) *
        //       ((Bxi + Bx[j]) * termA_avg[0] + (Byi + By[j]) * termA_avg[1] + (Bzi + Bz[j]) * termA_avg[2]); // GDSPH
        f_i += mj * ((Bxi * termA1_i + Byi * termA2_i + Bzi * termA3_i) * rhosqinv_i +
                     (Bx[j] * termA1_j + By[j] * termA2_j + Bz[j] * termA3_j) * rhosqinv_j); // PHANTOM

        momentum_x += mj * (rhosqinv_i * momentum_xi + rhosqinv_j * momentum_xj) - a_visc_x;
        momentum_y += mj * (rhosqinv_i * momentum_yi + rhosqinv_j * momentum_yj) - a_visc_y;
        momentum_z += mj * (rhosqinv_i * momentum_zi + rhosqinv_j * momentum_zj) - a_visc_z;
    }

    a_visc_energy = stl::max(T(0), a_visc_energy);
    Tc eCoeff     = (tdpdTrho == nullptr) ? pi / (kxi * mi * mi * gradhi) : tdpdTrho[i];
    du[i]         = K * (eCoeff * energy + T(0.5) * a_visc_energy); // factor of 2 already removed from 2P/rho

    // grad_P_xyz is stored as the acceleration,s accel = -grad_P / rho
    grad_P_x[i] = K * (momentum_x - Bxi * f_i * H * mu_0Inv);
    grad_P_y[i] = K * (momentum_y - Byi * f_i * H * mu_0Inv);
    grad_P_z[i] = K * (momentum_z - Bzi * f_i * H * mu_0Inv);
    *maxvsignal = maxvsignali;
}

} // namespace sph::magneto
