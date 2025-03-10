#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/sfc/box.hpp"
#include "cstone/traversal/ijloop/ijloop.hpp"

#include "sph/kernels.hpp"
#include "sph/table_lookup.hpp"

namespace sph
{

template<class T>
struct IADInteractionSTD
{
    const T* wh;

    template<class ParticleData, class Tc>
    auto operator()(const ParticleData& iData, const ParticleData& jData, cstone::Vec3<Tc> const& r_ij, T r2) const
    {
        const auto [i, iPos, hi, mi, roi] = iData;
        const auto [j, jPos, hj, mj, roj] = jData;

        T hiInv = T(1) / hi;

        T dist = std::sqrt(r2);

        T vloc = dist * hiInv;
        T w    = lt::lookup(wh, vloc);

        T mj_roj_w = mj / roj * w;

        return std::make_tuple(           //
            r_ij[0] * r_ij[0] * mj_roj_w, //
            r_ij[0] * r_ij[1] * mj_roj_w, //
            r_ij[0] * r_ij[2] * mj_roj_w, //
            r_ij[1] * r_ij[1] * mj_roj_w, //
            r_ij[1] * r_ij[2] * mj_roj_w, //
            r_ij[2] * r_ij[2] * mj_roj_w);
    }
};

template<class T, class Tc>
struct IADPostambleSTD
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

template<size_t stride = 1, class Tc, class Tm, class T>
HOST_DEVICE_FUN inline void IADJLoopSTD(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box,
                                        const cstone::LocalIndex* neighbors, unsigned neighborsCount, const Tc* x,
                                        const Tc* y, const Tc* z, const T* h, const Tm* m, const T* rho, const T* wh,
                                        const T* /*whd*/, T* c11, T* c12, T* c13, T* c22, T* c23, T* c33)
{
    IADInteractionSTD      interaction{wh};
    IADPostambleSTD<T, Tc> postamble{K};

    const auto input  = std::make_tuple(m, rho);
    const auto output = std::make_tuple(c11, c12, c13, c22, c23, c33);

    const auto iData  = cstone::ijloop::loadParticleData(x, y, z, h, input, i);
    const bool usePbc = cstone::ijloop::requiresPbcHandling(box, iData);

    decltype(interaction(iData, iData, cstone::Vec3<Tc>{}, T{})) result = {};

    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[stride * pj];

        const auto jData = cstone::ijloop::loadParticleData(x, y, z, h, input, j);

        const auto [r_ij, r2] = cstone::ijloop::posDiffAndDistSq(usePbc, box, iData, jData);

        cstone::ijloop::updateResult(result, interaction(iData, jData, r_ij, r2));
    }

    result = postamble(iData, result);

    cstone::ijloop::storeParticleData(output, i, result);
}

} // namespace sph
