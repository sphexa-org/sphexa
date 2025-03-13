#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/sfc/box.hpp"
#include "cstone/util/tuple_util.hpp"
#include "cstone/traversal/ijloop/ijloop.hpp"

#include "sph/kernels.hpp"
#include "sph/table_lookup.hpp"

namespace sph
{

template<class T>
struct MomentumAndEnergyInteractionStd
{
    const T* wh;

    template<class ParticleData, class Tc>
    constexpr auto operator()(const ParticleData& iData, const ParticleData& jData, cstone::Vec3<Tc> const& r_ij,
                              T r2) const
    {
        constexpr T gradh_i = 1.0;
        constexpr T gradh_j = 1.0;

        const auto [i, iPos, hi, mi, roi, vxi, vyi, vzi, pri, ci, c11i, c12i, c13i, c22i, c23i, c33i] = iData;
        const auto [j, jPos, hj, mj, roj, vxj, vyj, vzj, prj, cj, c11j, c12j, c13j, c22j, c23j, c33j] = jData;

        T rx = r_ij[0];
        T ry = r_ij[1];
        T rz = r_ij[2];

        T    hiInv  = T(1) / hi;
        T    hiInv3 = hiInv * hiInv * hiInv;
        auto mi_roi = mi / roi;

        T dist = std::sqrt(r2);

        T vx_ij = vxi - vxj;
        T vy_ij = vyi - vyj;
        T vz_ij = vzi - vzj;

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

        T termA1_j = c11j * rx + c12j * ry + c13j * rz;
        T termA2_j = c12j * rx + c22j * ry + c23j * rz;
        T termA3_j = c13j * rx + c23j * ry + c33j * rz;

        T           wij          = i == j ? 0 : rv / dist;
        constexpr T av_alpha     = T(1);
        T           viscosity_ij = T(0.5) * artificial_viscosity(av_alpha, av_alpha, ci, cj, wij);

        // For time-step calculations
        T vijsignal = i == j ? 0 : ci + cj - T(3) * wij;

        auto mj_roj_Wj = mj / roj * Wj;

        T mj_pro_i = mj * pri / (gradh_i * roi * roi);

        T momentum_x, momentum_y, momentum_z;
        {
            T a = Wi * (mj_pro_i + viscosity_ij * mi_roi);
            T b = mj_roj_Wj * (prj / (roj * gradh_j) + viscosity_ij);

            momentum_x = a * termA1_i + b * termA1_j;
            momentum_y = a * termA2_i + b * termA2_j;
            momentum_z = a * termA3_i + b * termA3_j;
        }
        T energy;
        {
            T a = Wi * (T(2) * mj_pro_i + viscosity_ij * mi_roi);
            T b = viscosity_ij * mj_roj_Wj;

            energy = vx_ij * (a * termA1_i + b * termA1_j) + vy_ij * (a * termA2_i + b * termA2_j) +
                     vz_ij * (a * termA3_i + b * termA3_j);
        }
        return std::make_tuple(energy, momentum_x, momentum_y, momentum_z, cstone::ijloop::reduction::max(vijsignal));
    }
};

template<class Tc>
struct MomentumAndEnergyPostambleStd
{
    Tc K;

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData&, const Result& result) const
    {
        const auto [energy, momentum_x, momentum_y, momentum_z, maxvsignal] = result;
        // with the choice of calculating coordinate (r) and velocity (v_ij) differences as i - j,
        // we add the negative sign only here at the end instead of to termA123_ij in each interaction
        return std::make_tuple(-K * Tc(0.5) * energy, K * momentum_x, K * momentum_y, K * momentum_z, maxvsignal);
    };
};

template<class Tc>
struct MomentumAndEnergyPostambleStdWithDt : MomentumAndEnergyPostambleStd<Tc>
{
    Tc Kcour;

    MomentumAndEnergyPostambleStdWithDt(Tc K, Tc Kcour)
        : MomentumAndEnergyPostambleStd<Tc>{K}
        , Kcour(Kcour)
    {
    }

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        const auto [du, grad_P_x, grad_P_y, grad_P_z, maxvsignal] =
            MomentumAndEnergyPostambleStd<Tc>::operator()(iData, result);
        const auto [i, iPos, hi, mi, roi, vxi, vyi, vzi, pri, ci, c11i, c12i, c13i, c22i, c23i, c33i] = iData;

        auto dt = tsKCourant(maxvsignal.value, hi, ci, Kcour);
        return std::make_tuple(du, grad_P_x, grad_P_y, grad_P_z, dt);
    };
};

template<size_t stride = 1, class Tc, class Tm, class T, class Tm1>
HOST_DEVICE_FUN inline void
momentumAndEnergyJLoop(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box, const cstone::LocalIndex* neighbors,
                       unsigned neighborsCount, const Tc* x, const Tc* y, const Tc* z, const T* vx, const T* vy,
                       const T* vz, const T* h, const Tm* m, const T* rho, const T* p, const T* c, const T* c11,
                       const T* c12, const T* c13, const T* c22, const T* c23, const T* c33, const T* wh,
                       const T* /*whd*/, T* grad_P_x, T* grad_P_y, T* grad_P_z, Tm1* du, T* maxvsignal)
{
    MomentumAndEnergyInteractionStd interaction{wh};
    MomentumAndEnergyPostambleStd   postamble{K};

    const auto input  = std::make_tuple(m, rho, vx, vy, vz, p, c, c11, c12, c13, c22, c23, c33);
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

    result = postamble(iData, result);

    cstone::ijloop::storeParticleData(output, i, result);
}

template<class Neighborhood, class Tc, class T, class Tm, class Tm1>
void momentumAndEnergyIjLoop(Neighborhood const& neighborhood, Tc K, Tc Kcour, const Tm* m, const T* rho, const T* vx,
                             const T* vy, const T* vz, const T* p, const T* c, const T* c11, const T* c12, const T* c13,
                             const T* c22, const T* c23, const T* c33, const T* wh, Tm1* du, T* grad_P_x, T* grad_P_y,
                             T* grad_P_z, T* dt)
{
    neighborhood.ijLoop(std::make_tuple(m, rho, vx, vy, vz, p, c, c11, c12, c13, c22, c23, c33),
                        std::make_tuple(du, grad_P_x, grad_P_y, grad_P_z, dt), MomentumAndEnergyInteractionStd{wh},
                        MomentumAndEnergyPostambleStdWithDt{K, Kcour},
                        cstone::ijloop::symmetry::asymmetric); // TODO: different symmetries per output
}

} // namespace sph
