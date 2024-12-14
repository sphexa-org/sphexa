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
 * @brief Barnes-Hut tree walk with Ewald periodic boundaries
 *
 * @author Jonathan Coles <jonathan.coles@cscs.ch>
 */

#pragma once

#include "cstone/traversal/traversal.hpp"
#include "cstone/traversal/macs.hpp"
#include "cstone/focus/source_center.hpp"
#include "cartesian_qpole.hpp"
#include "ewald.h"

namespace ryoanji
{

/*! @brief Coefficients used by the sum over k-space.
 *
 */
template<class T1>
struct EwaldHSumCoefficients
{
    Vec3<T1> hr;
    Vec3<T1> hr_scaled;
    T1       hfac_cos;
    T1       hfac_sin;
};

/*! @brief Parameters used by the real- and k-space routines.
 *
 */
template<class Tc, class Tmm>
class EwaldParameters
{
    //! @brief return size of hsum_coeffs for a given value of ceil(hCut)
    HOST_DEVICE_FUN constexpr static int hsumCoeffSize(int ceilHcut)
    {
        int numOneDim = (2 * ceilHcut + 1);
        return numOneDim * numOneDim * numOneDim;
    }

    //! @brief max supported value of ceil(hCut)
    constexpr static inline int maxCeilHcut = 3;

public:
    //! @brief return number of hsum_coeffs
    HOST_DEVICE_FUN int numHsumCoeffs() const
    {
        auto ceilHcut = int(std::ceil(hCut));
        assert(ceilHcut <= maxCeilHcut);
        return hsumCoeffSize(ceilHcut);
    }

    CartesianQuadrupole<Tmm> Mroot;
    Vec3<Tc>                 Mroot_center;
    int                      numReplicaShells;
    int                      numEwaldShells;
    double                   L;
    double                   lCut;
    double                   hCut;
    double                   alpha_scale;
    double                   small_R_scale_factor;

    util::array<EwaldHSumCoefficients<Tc>, hsumCoeffSize(maxCeilHcut)> hsum_coeffs;
};

template<class T>
using CartesianQuadrupoleGamma = util::array<T, 6>;

/*! @brief Evaluate the potential and acceleration of a multipole expansion.
 *
 * @tparam        T1         float or double
 * @tparam        T2         float or double
 * @param[in]     r          target x,y,z coordinates relative to multipole center
 * @param[in]     gamma      multipole evaluation coefficients
 * @param[in]     multipole  multipole source
 * @return                   tupole(u,ax,ay,az)
 *
 */
template<class Ta, class Tc, class Tmm>
HOST_DEVICE_FUN Vec4<Ta> ewaldEvalMultipoleComplete(Vec4<Ta> potAcc, Vec3<Tc> r_tc,
                                                    const CartesianQuadrupoleGamma<Tmm>& gamma,
                                                    const CartesianQuadrupole<Tmm>&      multipole)
{
    const Vec3<Ta> r{Ta(r_tc[0]), Ta(r_tc[1]), Ta(r_tc[2])};

    const Tmm M   = multipole[Cqi::mass];
    const Tmm qxx = (multipole[Cqi::qxx] + multipole[Cqi::trace]) / Tmm(3);
    const Tmm qyy = (multipole[Cqi::qyy] + multipole[Cqi::trace]) / Tmm(3);
    const Tmm qzz = (multipole[Cqi::qzz] + multipole[Cqi::trace]) / Tmm(3);
    const Tmm qxy = (multipole[Cqi::qxy]) / Tmm(3);
    const Tmm qxz = (multipole[Cqi::qxz]) / Tmm(3);
    const Tmm qyz = (multipole[Cqi::qyz]) / Tmm(3);

    Vec3<Ta> Qr{Ta(r[0] * qxx + r[1] * qxy + r[2] * qxz), Ta(r[0] * qxy + r[1] * qyy + r[2] * qyz),
                Ta(r[0] * qxz + r[1] * qyz + r[2] * qzz)};

    Ta rQr = 0.5 * dot(r, Qr);
    Ta Qtr = 0.5 * multipole[Cqi::trace];

    Ta       ugrav = -gamma[0] * M + gamma[1] * Qtr - gamma[2] * rQr;
    Vec3<Ta> a     = gamma[2] * Qr - r * (+gamma[1] * M - gamma[2] * Qtr + gamma[3] * rQr);

    return potAcc + Vec4<Ta>{ugrav, a[0], a[1], a[2]};
}

/*! @brief Compute Ewald coefficients for k-space summation.
 *
 * @tparam        Tc                float or double, real-space coordinates
 * @tparam        Tmm               float or double, multipole moments
 * @param[in]     Mroot             top-level box multipole
 * @param[in]     Mroot_center      center of the multipole expansion
 * @param[in]     numReplicaShells  number of replica shells by tree gravity
 * @param[in]     L                 box side-length
 * @param[in]     lCut              cutoff distance for Ewald l sum (real space) (recommended: 2.6)
 * @param[in]     hCut              cutoff distance for Ewald h sum (k-space) (recommended: 2.8)
 * @param[in]     alpha_scale       scale factor for the mixing factor alpha (recommended: 2.0)
 * @return                          the initialized EwaldParameters struct
 *
 * Evaluates the monopole of the given multipole expansion at the position
 * (rx,ry,rz) relative to the multipole center.
 */
template<class Tc, class Tmm>
EwaldParameters<Tc, Tmm> ewaldInitParameters(const CartesianQuadrupole<Tmm>& Mroot, const Vec3<Tc>& Mroot_center,
                                             int numReplicaShells, double L, double lCut, double hCut,
                                             double alpha_scale, double smallR)
{
    // This will disable Ewald, but still allow a normal gravity calculation using replicas.
    if (lCut == 0 && hCut == 0 && alpha_scale == 0) { numReplicaShells = 0; }

    EwaldParameters<Tc, Tmm> params = {.Mroot                = Mroot,
                                       .Mroot_center         = Mroot_center,
                                       .numReplicaShells     = numReplicaShells,
                                       .numEwaldShells       = std::max(int(ceil(lCut)), numReplicaShells),
                                       .L                    = L,
                                       .lCut                 = lCut,
                                       .hCut                 = hCut,
                                       .alpha_scale          = alpha_scale,
                                       .small_R_scale_factor = smallR,
                                       .hsum_coeffs          = {}};

    if (params.numEwaldShells == 0) { return params; }

    const auto hReps = int(std::ceil(hCut));
    const auto alpha = alpha_scale / L;
    const auto k4    = M_PI * M_PI / (alpha * alpha * L * L);
    const auto hCut2 = hCut * hCut;

    int pushBackCnt = 0;
    for (auto hx = -hReps; hx <= hReps; hx++)
    {
        for (auto hy = -hReps; hy <= hReps; hy++)
        {
            for (auto hz = -hReps; hz <= hReps; hz++)
            {
                Vec3<Tc> hr = {Tc(hx), Tc(hy), Tc(hz)};

                const auto h2 = norm2(hr);
                if (h2 == 0) continue;
                if (h2 > hCut2) continue;

                const Tmm g0 = exp(-k4 * h2) / (M_PI * h2 * L);
                const Tmm g1 = 2 * M_PI / L * g0;
                const Tmm g2 = -2 * M_PI / L * g1;
                const Tmm g3 = 2 * M_PI / L * g2;
                const Tmm g4 = -2 * M_PI / L * g3;
                const Tmm g5 = 2 * M_PI / L * g4;

                CartesianQuadrupoleGamma<Tmm> gamma1{g0, 0.0, g2, 0.0, g4, 0.0};
                const auto mfac_cos = ewaldEvalMultipoleComplete(Vec4<Tmm>{0, 0, 0, 0}, hr, gamma1, Mroot)[0];

                CartesianQuadrupoleGamma<Tmm> gamma2{0.0, g1, 0.0, g3, 0.0, g5};
                const auto mfac_sin = ewaldEvalMultipoleComplete(Vec4<Tmm>{0, 0, 0, 0}, hr, gamma2, Mroot)[0];

                EwaldHSumCoefficients<Tc> hsum = {
                    .hr        = hr,
                    .hr_scaled = 2 * M_PI / L * hr,
                    .hfac_cos  = mfac_cos,
                    .hfac_sin  = mfac_sin,
                };

                params.hsum_coeffs[pushBackCnt++] = hsum;
            }
        }
    }

    return params;
}

//! @brief real space Ewald contribution to the potential and acceleration of a single particle @p r
template<class T1, class T2>
HOST_DEVICE_FUN Vec4<T1> computeEwaldRealSpace(Vec3<T1> r, const EwaldParameters<T1, T2>& params)
{
    const auto Mroot            = params.Mroot;
    const T1   lCut             = params.lCut;
    const T1   alpha_scale      = params.alpha_scale;
    const T1   L                = params.L;
    const int  numEwaldShells   = params.numEwaldShells;
    const int  numReplicaShells = params.numReplicaShells;

    const T1 lCut2    = lCut * lCut * L * L;
    const T1 alpha    = alpha_scale / L;
    const T1 alpha2   = alpha * alpha;
    const T1 k1       = M_PI / (alpha2 * L * L * L);
    const T1 ka       = 2.0 * alpha / sqrt(M_PI);
    const T1 small_R2 = params.small_R_scale_factor * L * L;

    Vec4<T1> potAcc{k1 * Mroot[Cqi::mass], 0, 0, 0};

    r -= params.Mroot_center;

    for (auto ix = -numEwaldShells; ix <= numEwaldShells; ix++)
    {
        for (auto iy = -numEwaldShells; iy <= numEwaldShells; iy++)
        {
            for (auto iz = -numEwaldShells; iz <= numEwaldShells; iz++)
            {
                CartesianQuadrupoleGamma<T2> gamma;

                //
                // This is the region that was computed with the normal gravity routine.
                // We need to know this below to add a correction.
                //
                const auto in_precomputed_region = (-numReplicaShells <= ix && ix <= numReplicaShells) &&
                                                   (-numReplicaShells <= iy && iy <= numReplicaShells) &&
                                                   (-numReplicaShells <= iz && iz <= numReplicaShells);

                Vec3<T1> Rrep{ix * L, iy * L, iz * L};

                const auto R  = r + Rrep;
                const auto R2 = norm2(R);

                // in other words: !(r2 <= lCut2 || in_precomputed_region)
                if (R2 > lCut2 && !in_precomputed_region) continue;

                bool use_small_r_approximation = R2 < small_R2 && ka > 0;
                if (use_small_r_approximation)
                {
                    //
                    // For small r, series expand about the origin to avoid errors
                    // caused by cancellation of large terms.
                    //
                    auto c0   = ka;
                    auto R2a2 = R2 * alpha2;

                    gamma[0] = c0 * (R2a2 / 3.0 - 1.0);
                    c0 *= 2 * alpha2;
                    gamma[1] = c0 * (R2a2 / 5.0 - 1.0 / 3.0);
                    c0 *= 2 * alpha2;
                    gamma[2] = c0 * (R2a2 / 7.0 - 1.0 / 5.0);
                    c0 *= 2 * alpha2;
                    gamma[3] = c0 * (R2a2 / 9.0 - 1.0 / 7.0);
                    c0 *= 2 * alpha2;
                    gamma[4] = c0 * (R2a2 / 11.0 - 1.0 / 9.0);
                    c0 *= 2 * alpha2;
                    gamma[5] = c0 * (R2a2 / 13.0 - 1.0 / 11.0);
                }
                else
                {
                    const auto Rmag  = sqrt(R2);
                    const auto invR  = 1.0 / Rmag;
                    const auto invR2 = invR * invR;
                    const auto a     = exp(-R2 * alpha2) * ka * invR2;

                    auto alphan = T1(1.0);

                    //
                    // If we are in the region already computed by the normal gravity routine
                    // then only add a -erf correction, otherwise we need the full erfc value.
                    //
                    auto fn = in_precomputed_region ? -erf(alpha * Rmag) : erfc(alpha * Rmag);

                    gamma[0] = fn * invR;
                    gamma[1] = gamma[0] * invR2 + a;
                    alphan *= 2 * alpha2;
                    gamma[2] = 3 * gamma[1] * invR2 + alphan * a;
                    alphan *= 2 * alpha2;
                    gamma[3] = 5 * gamma[2] * invR2 + alphan * a;
                    alphan *= 2 * alpha2;
                    gamma[4] = 7 * gamma[3] * invR2 + alphan * a;
                    alphan *= 2 * alpha2;
                    gamma[5] = 9 * gamma[4] * invR2 + alphan * a;
                }

                potAcc = ewaldEvalMultipoleComplete(potAcc, R, gamma, Mroot);
            }
        }
    }

    return potAcc;
}

//! @brief Ewald K-space contribution to the potential and acceleration of a single particle @p r
template<class T1, class T2>
HOST_DEVICE_FUN Vec4<T1> computeEwaldKSpace(Vec3<T1> r, const EwaldParameters<T1, T2>& params)
{
    Vec4<T1> potAcc = Vec4<T1>{0, 0, 0, 0};
    Vec3<T1> dr     = r - params.Mroot_center;

    for (int i = 0; i < params.numHsumCoeffs(); ++i)
    {
        const auto& hd = params.hsum_coeffs[i];

        auto hdotx   = dot(hd.hr_scaled, dr);
        auto c       = std::cos(hdotx);
        auto s       = std::sin(hdotx);
        auto cs_sum  = hd.hfac_cos * c + hd.hfac_sin * s;
        auto cs_diff = hd.hfac_cos * s - hd.hfac_sin * c;

        potAcc[0] -= cs_sum;
        potAcc[1] += cs_diff * hd.hr_scaled[0];
        potAcc[2] += cs_diff * hd.hr_scaled[1];
        potAcc[3] += cs_diff * hd.hr_scaled[2];
    }

    return potAcc;
}

/*! @brief CPU driver for gravitational corrections using Ewald summation
 *
 * @tparam KeyType               unsigned 32- or 64-bit integer type
 * @tparam Tc                    float or double
 * @tparam Ta                    float or double
 * @param[in]    rootCenter      location of root cell center of mass
 * @param[in]    multipoles      array of length @p octree.numTreeNodes() with the multipole moments for all nodes
 * @param[in]    layout          array of length @p octree.numLeafNodes()+1, layout[i] is the start offset
 *                               into the x,y,z,m arrays for the leaf node with index i. The last element
 *                               is equal to the length of the x,y,z,m arrays.
 * @param[in]    firstTarget
 * @param[in]    lastTarget
 * @param[in]    x               x-coordinates
 * @param[in]    y               y-coordinates
 * @param[in]    z               z-coordinates
 * @param[in]    m               masses
 * @param[in]    G               gravitational constant
 * @param[inout] ugrav           location to add potential per particle to, can be nullptr
 * @param[inout] ax              location to add x-acceleration to
 * @param[inout] ay              location to add y-acceleration to
 * @param[inout] az              location to add z-acceleration to
 * @param[inout] ugravTot        total gravitational energy, one element
 * @param[in]    numReplicaShells  see @a ewaldInitParameters
 * @param[in]    lCut              see @a ewaldInitParameters
 * @param[in]    hCut              see @a ewaldInitParameters
 * @param[in]    alpha_scale       see @a ewaldInitParameters
 */
template<class MType, class Tc, class Ta, class Tm, class Tu>
void computeGravityEwald(const cstone::Vec3<Tc>& rootCenter, const MType& Mroot, LocalIndex firstTarget,
                         LocalIndex lastTarget, const Tc* x, const Tc* y, const Tc* z, const Tm* m,
                         const cstone::Box<Tc>& box, float G, Ta* ugrav, Ta* ax, Ta* ay, Ta* az, Tu* ugravTot,
                         EwaldSettings settings)
{
    if (box.minExtent() != box.maxExtent()) { throw std::runtime_error("Ewald gravity requires cubic bounding boxes"); }

    EwaldParameters<Tc, typename MType::value_type> ewaldParams =
        ewaldInitParameters(Mroot, rootCenter, settings.numReplicaShells, box.lx(), settings.lCut, settings.hCut,
                            settings.alpha_scale, settings.small_R_scale_factor);

    if (ewaldParams.numEwaldShells == 0) { return; }

    Tu Uewald = 0;
#pragma omp parallel for reduction(+ : Uewald)
    for (LocalIndex i = firstTarget; i < lastTarget; i++)
    {
        Vec3<Tc> target{x[i], y[i], z[i]};
        Vec4<Tc> potAcc{0, 0, 0, 0};

        potAcc += computeEwaldRealSpace(target, ewaldParams);
        potAcc += computeEwaldKSpace(target, ewaldParams);

        if (ugrav) { ugrav[i] += G * potAcc[0] * m[i]; }

        Uewald += potAcc[0] * m[i];
        ax[i] += G * potAcc[1];
        ay[i] += G * potAcc[2];
        az[i] += G * potAcc[3];
    }

    *ugravTot += 0.5 * G * Uewald;
}

} // namespace ryoanji
