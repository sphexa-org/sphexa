#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/sfc/box.hpp"
#include "cstone/util/tuple_util.hpp"

#include "sph/kernels.hpp"
#include "sph/table_lookup.hpp"

namespace sph
{

template<size_t stride = 1, class Tc, class Tm, class T, class Tm1>
HOST_DEVICE_FUN inline void
momentumAndEnergyJLoop(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box, const cstone::LocalIndex* neighbors,
                       unsigned neighborsCount, const Tc* x, const Tc* y, const Tc* z, const T* vx, const T* vy,
                       const T* vz, const T* h, const Tm* m, const T* rho, const T* p, const T* c, const T* c11,
                       const T* c12, const T* c13, const T* c22, const T* c23, const T* c33, const T* wh,
                       const T* /*whd*/, T* grad_P_x, T* grad_P_y, T* grad_P_z, Tm1* du, T* maxvsignal)
{
    constexpr T gradh_i = 1.0;
    constexpr T gradh_j = 1.0;

    auto xi  = x[i];
    auto yi  = y[i];
    auto zi  = z[i];
    auto vxi = vx[i];
    auto vyi = vy[i];
    auto vzi = vz[i];

    auto hi  = h[i];
    auto roi = rho[i];
    auto pri = p[i];
    auto ci  = c[i];

    auto mi_roi = m[i] / rho[i];

    T hiInv  = T(1) / hi;
    T hiInv3 = hiInv * hiInv * hiInv;

    T maxvsignali = 0.0;
    T momentum_x = 0.0, momentum_y = 0.0, momentum_z = 0.0, energy = 0.0;

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

        T vx_ij = vxi - vx[j];
        T vy_ij = vyi - vy[j];
        T vz_ij = vzi - vz[j];

        T hj    = h[j];
        T hjInv = T(1) / hj;

        T v1 = dist * hiInv;
        T v2 = dist * hjInv;

        T rv = rx * vx_ij + ry * vy_ij + rz * vz_ij;

        T hjInv3 = hjInv * hjInv * hjInv;
        T Wi     = hiInv3 * lt::lookup(wh, v1);
        T Wj     = hjInv3 * lt::lookup(wh, v2);

        T termA1_i = c11i * rx + c12i * ry + c13i * rz;
        T termA2_i = c12i * rx + c22i * ry + c23i * rz;
        T termA3_i = c13i * rx + c23i * ry + c33i * rz;

        auto c11j = c11[j];
        auto c12j = c12[j];
        auto c13j = c13[j];
        auto c22j = c22[j];
        auto c23j = c23[j];
        auto c33j = c33[j];

        T termA1_j = c11j * rx + c12j * ry + c13j * rz;
        T termA2_j = c12j * rx + c22j * ry + c23j * rz;
        T termA3_j = c13j * rx + c23j * ry + c33j * rz;

        auto roj = rho[j];
        auto cj  = c[j];

        T           wij          = rv / dist;
        constexpr T av_alpha     = T(1);
        T           viscosity_ij = T(0.5) * artificial_viscosity(av_alpha, av_alpha, ci, cj, wij);

        // For time-step calculations
        T vijsignal = ci + cj - T(3) * wij;
        maxvsignali = (vijsignal > maxvsignali) ? vijsignal : maxvsignali;

        auto mj        = m[j];
        auto mj_roj_Wj = mj / roj * Wj;

        T mj_pro_i = mj * pri / (gradh_i * roi * roi);

        {
            T a = Wi * (mj_pro_i + viscosity_ij * mi_roi);
            T b = mj_roj_Wj * (p[j] / (roj * gradh_j) + viscosity_ij);

            momentum_x += a * termA1_i + b * termA1_j;
            momentum_y += a * termA2_i + b * termA2_j;
            momentum_z += a * termA3_i + b * termA3_j;
        }
        {
            T a = Wi * (T(2) * mj_pro_i + viscosity_ij * mi_roi);
            T b = viscosity_ij * mj_roj_Wj;

            energy += vx_ij * (a * termA1_i + b * termA1_j) + vy_ij * (a * termA2_i + b * termA2_j) +
                      vz_ij * (a * termA3_i + b * termA3_j);
        }
    }

    // with the choice of calculating coordinate (r) and velocity (v_ij) differences as i - j,
    // we add the negative sign only here at the end instead of to termA123_ij in each iteration
    du[i]       = -K * Tm1(0.5) * energy;
    grad_P_x[i] = K * momentum_x;
    grad_P_y[i] = K * momentum_y;
    grad_P_z[i] = K * momentum_z;
    *maxvsignal = maxvsignali;
}

} // namespace sph
