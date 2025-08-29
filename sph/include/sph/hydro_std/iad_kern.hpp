#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/sfc/box.hpp"

#include "sph/kernels.hpp"
#include "sph/table_lookup.hpp"

namespace sph
{

template<size_t stride = 1, class Tc, class Tm, class T>
HOST_DEVICE_FUN inline void IADJLoopSTD(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box,
                                        const cstone::LocalIndex* neighbors, unsigned neighborsCount, const Tc* x,
                                        const Tc* y, const Tc* z, const T* h, const Tm* m, const T* rho, const T* wh,
                                        const T* /*whd*/, T* c11, T* c12, T* c13, T* c22, T* c23, T* c33)
{
    T tau11 = 0.0, tau12 = 0.0, tau13 = 0.0, tau22 = 0.0, tau23 = 0.0, tau33 = 0.0;

    auto xi = x[i];
    auto yi = y[i];
    auto zi = z[i];

    auto hi    = h[i];
    auto hiInv = T(1) / hi;

    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[stride * pj];

        T rx = (xi - x[j]);
        T ry = (yi - y[j]);
        T rz = (zi - z[j]);

        applyPBC(box, T(2) * hi, rx, ry, rz);

        T dist = std::sqrt(rx * rx + ry * ry + rz * rz);

        T vloc = dist * hiInv;
        T w    = lt::lookup(wh, vloc);

        T mj_roj_w = m[j] / rho[j] * w;

        tau11 += rx * rx * mj_roj_w;
        tau12 += rx * ry * mj_roj_w;
        tau13 += rx * rz * mj_roj_w;
        tau22 += ry * ry * mj_roj_w;
        tau23 += ry * rz * mj_roj_w;
        tau33 += rz * rz * mj_roj_w;
    }

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

    c11[i] = (tau22 * tau33 - tau23 * tau23) * factor;
    c12[i] = (tau13 * tau23 - tau33 * tau12) * factor;
    c13[i] = (tau12 * tau23 - tau22 * tau13) * factor;
    c22[i] = (tau11 * tau33 - tau13 * tau13) * factor;
    c23[i] = (tau13 * tau12 - tau11 * tau23) * factor;
    c33[i] = (tau11 * tau22 - tau12 * tau12) * factor;
}
} // namespace sph
