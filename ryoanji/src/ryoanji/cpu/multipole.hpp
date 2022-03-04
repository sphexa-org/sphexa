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
 * @brief implements elementary gravity data structures for octree nodes
 *
 * @author Sebastian Keller        <sebastian.f.keller@gmail.com>
 *
 * See for example Hernquist 1987, Performance Characteristics of Tree Codes,
 * https://ui.adsabs.harvard.edu/abs/1987ApJS...64..715H
 */

#pragma once

#include <cmath>

#include "cstone/util/tuple.hpp"
#include "ryoanji/types.h"

namespace ryoanji
{

template <class T>
using CartesianQuadrupole = util::array<T, 8>;

template<class MType>
struct IsCartesian : public stl::integral_constant<size_t, MType{}.size() == CartesianQuadrupole<float>{}.size()>
{
};

//! @brief CartesianQuadrupole index names
struct Cqi
{
    enum IndexNames
    {
        mass = 0,
        qxx  = 1,
        qxy  = 2,
        qxz  = 3,
        qyy  = 4,
        qyz  = 5,
        qzz  = 6
    };
};

/*! @brief Compute the monopole and quadruple moments from particle coordinates
 *
 * @tparam       T1             float or double
 * @tparam       T2             float or double
 * @tparam       T3             float or double
 * @param[in]    x              x coordinate array
 * @param[in]    y              y coordinate array
 * @param[in]    z              z coordinate array
 * @param[in]    m              masses array
 * @param[in]    numParticles   number of particles to read from coordinate arrays
 * @param[out]   gv             output quadrupole
 */
template<class T1, class T2, class T3>
void particle2Multipole(const T1* x,
                        const T1* y,
                        const T1* z,
                        const T2* m,
                        LocalIndex first,
                        LocalIndex last,
                        const Vec3<T1>& center,
                        CartesianQuadrupole<T3>& gv)
{
    gv = T3(0);
    if (first == last) { return; }

    for (LocalIndex i = first; i < last; ++i)
    {
        T1 xx  = x[i];
        T1 yy  = y[i];
        T1 zz  = z[i];
        T1 m_i = m[i];

        T1 rx = xx - center[0];
        T1 ry = yy - center[1];
        T1 rz = zz - center[2];

        gv[Cqi::mass] += m_i;
        gv[Cqi::qxx] += rx * rx * m_i;
        gv[Cqi::qxy] += rx * ry * m_i;
        gv[Cqi::qxz] += rx * rz * m_i;
        gv[Cqi::qyy] += ry * ry * m_i;
        gv[Cqi::qyz] += ry * rz * m_i;
        gv[Cqi::qzz] += rz * rz * m_i;
    }

    T1 traceQ = gv[Cqi::qxx] + gv[Cqi::qyy] + gv[Cqi::qzz];

    // remove trace
    gv[Cqi::qxx] = 3 * gv[Cqi::qxx] - traceQ;
    gv[Cqi::qyy] = 3 * gv[Cqi::qyy] - traceQ;
    gv[Cqi::qzz] = 3 * gv[Cqi::qzz] - traceQ;
    gv[Cqi::qxy] *= 3;
    gv[Cqi::qxz] *= 3;
    gv[Cqi::qyz] *= 3;
}

template<class T>
void moveExpansionCenter(Vec3<T> Xold, Vec3<T> Xnew, CartesianQuadrupole<T>& gv)
{
    Vec3<T> dX = Xold - Xnew;
    T rx = dX[0];
    T ry = dX[1];
    T rz = dX[2];

    gv[Cqi::qxx] = gv.qxx - rx * rx * gv[Cqi::mass];
    gv[Cqi::qxy] = gv.qxy - rx * ry * gv[Cqi::mass];
    gv[Cqi::qxz] = gv.qxz - rx * rz * gv[Cqi::mass];
    gv[Cqi::qyy] = gv.qyy - ry * ry * gv[Cqi::mass];
    gv[Cqi::qyz] = gv.qyz - ry * rz * gv[Cqi::mass];
    gv[Cqi::qzz] = gv.qzz - rz * rz * gv[Cqi::mass];

    T traceQ = gv[Cqi::qxx] + gv[Cqi::qyy] + gv[Cqi::qzz];

    // remove trace
    gv[Cqi::qxx] = 3 * gv[Cqi::qxx] - traceQ;
    gv[Cqi::qyy] = 3 * gv[Cqi::qyy] - traceQ;
    gv[Cqi::qzz] = 3 * gv[Cqi::qzz] - traceQ;
    gv[Cqi::qxy] *= 3;
    gv[Cqi::qxz] *= 3;
    gv[Cqi::qyz] *= 3;
}

/*! @brief compute a single particle-particle gravitational interaction
 *
 * @tparam T1   float or double
 * @tparam T2   float or double
 * @param tx    target x coord
 * @param ty    target y coord
 * @param tz    target z coord
 * @param th    target h smoothing length
 * @param sx    source x coord
 * @param sy    source y coord
 * @param sz    source z coord
 * @param sh    source h coord
 * @param sm    source mass
 * @return      tuple(ax, ay, az, ugrav)
 */
template<class T1, class T2>
HOST_DEVICE_FUN inline __attribute__((always_inline)) util::tuple<T1, T1, T1, T1>
particle2Particle(T1 tx, T1 ty, T1 tz, T2 th, T1 sx, T1 sy, T1 sz, T2 sh, T2 sm)
{
    T1 rx = sx - tx;
    T1 ry = sy - ty;
    T1 rz = sz - tz;

    T1 r_2 = rx * rx + ry * ry + rz * rz;
    T1 r = std::sqrt(r_2);
    T1 r_minus1 = 1.0 / r;
    T1 r_minus2 = r_minus1 * r_minus1;

    T1 mEffective = sm;

    T1 h_ts = th + sh;
    if (r < h_ts)
    {
        // apply mass softening correction
        T1 vgr = r / h_ts;
        mEffective *= vgr * vgr * vgr;
    }

    T1 Mr_minus3 = mEffective * r_minus1 * r_minus2;

    return {Mr_minus3 * rx, Mr_minus3 * ry, Mr_minus3 * rz, -Mr_minus3 * r_2};
}

/*! @brief direct gravity calculation with particle-particle interactions
 *
 * @tparam       T1          float or double
 * @tparam       T2          float or double
 * @param[in]    tx          target particle x coordinate
 * @param[in]    ty          target particle y coordinate
 * @param[in]    tz          target particle z coordinate
 * @param[in]    hi          target particle smoothing length
 * @param[in]    sx          source particle x coordinates
 * @param[in]    sy          source particle y coordinates
 * @param[in]    sz          source particle z coordinates
 * @param[in]    h           source particle smoothing lengths
 * @param[in]    m           source particle masses
 * @param[in]    numSources  number of source particles
 * @return                   tuple(ax, ay, az, ugrav)
 *
 * Computes direct particle-particle gravitational interaction according to
 *
 *      vec(a_t) = - sum_{j} m_j / (r_tj^2 + eps2)^(3/2)) * (r_t - r_j)
 *
 * Notes:
 *  - Source particles MUST NOT contain the target. If the source is a cell that contains the target,
 *    the target must be located and this function called twice, with all particles before target and
 *    all particles that follow it.
 */
template<class T1, class T2>
HOST_DEVICE_FUN util::tuple<T1, T1, T1, T1> particle2Particle(T1 tx,
                                                              T1 ty,
                                                              T1 tz,
                                                              T2 hi,
                                                              const T1* sx,
                                                              const T1* sy,
                                                              const T1* sz,
                                                              const T2* h,
                                                              const T2* m,
                                                              LocalIndex numSources)
{
    T1 axLoc = 0;
    T1 ayLoc = 0;
    T1 azLoc = 0;
    T1 uLoc  = 0;

    #if defined(__llvm__) || defined(__clang__)
        #pragma clang loop vectorize(enable)
    #endif
    for (LocalIndex j = 0; j < numSources; ++j)
    {
        auto [ax_, ay_, az_, u_] = particle2Particle(tx, ty, tz, hi, sx[j], sy[j], sz[j], h[j], m[j]);

        axLoc += ax_;
        ayLoc += ay_;
        azLoc += az_;
        uLoc += u_;
    }

    return {axLoc, ayLoc, azLoc, uLoc};
}

/*! @brief apply gravitational interaction with a multipole to a particle
 *
 * @tparam        T1         float or double
 * @tparam        T2         float or double
 * @param[in]     tx         target particle x coordinate
 * @param[in]     ty         target particle y coordinate
 * @param[in]     tz         target particle z coordinate
 * @param[in]     multipole  multipole source
 * @param[inout]  ugrav      location to add gravitational potential to
 * @return                   tuple(ax, ay, az, u)
 *
 * Note: contribution is added to output
 *
 * Direct implementation of the formulae in Hernquist, 1987 (complete reference in file docstring):
 *
 * monopole:   -M/r^3 * vec(r)
 * quadrupole: Q*vec(r) / r^5 - 5/2 * vec(r)*Q*vec(r) * vec(r) / r^7
 */
template<class T1, class T2>
HOST_DEVICE_FUN inline util::tuple<T1, T1, T1, T1> multipole2Particle(T1 tx, T1 ty, T1 tz, const Vec3<T1>& center,
                                                                      const CartesianQuadrupole<T2>& multipole)
{
    T2 rx = tx - center[0];
    T2 ry = ty - center[1];
    T2 rz = tz - center[2];

    T2 r_2      = rx * rx + ry * ry + rz * rz;
    T2 r_minus1 = T2(1) / std::sqrt(r_2);
    T2 r_minus2 = r_minus1 * r_minus1;
    T2 r_minus5 = r_minus2 * r_minus2 * r_minus1;

    T2 Qrx = rx * multipole[Cqi::qxx] + ry * multipole[Cqi::qxy] + rz * multipole[Cqi::qxz];
    T2 Qry = rx * multipole[Cqi::qxy] + ry * multipole[Cqi::qyy] + rz * multipole[Cqi::qyz];
    T2 Qrz = rx * multipole[Cqi::qxz] + ry * multipole[Cqi::qyz] + rz * multipole[Cqi::qzz];

    T2 rQr = rx * Qrx + ry * Qry + rz * Qrz;
    //                  rQr quad-term           mono-term
    //                      |                     |
    T2 rQrAndMonopole = (T2(-2.5) * rQr * r_minus5 - multipole[Cqi::mass] * r_minus1) * r_minus2;

    //       Qr Quad-term
    return {r_minus5 * Qrx + rQrAndMonopole * rx,
            r_minus5 * Qry + rQrAndMonopole * ry,
            r_minus5 * Qrz + rQrAndMonopole * rz,
            -(multipole[Cqi::mass] * r_minus1 + T2(0.5) * r_minus5 * rQr)};
}

/*! @brief add a multipole contribution to the composite multipole
 *
 * @tparam        T           float or double
 * @param[inout]  composite   the composite multipole
 * @param[in]     dX          distance vector between composite and added expansion center
 * @param[in]     addend      the multipole to add
 *
 * Implements formula (2.5) from Hernquist 1987 (parallel axis theorem)
 */
template<class T>
void addQuadrupole(CartesianQuadrupole<T>& composite, Vec3<T> dX, const CartesianQuadrupole<T>& addend)
{
    T rx = dX[0];
    T ry = dX[1];
    T rz = dX[2];

    T rx_2 = rx * rx;
    T ry_2 = ry * ry;
    T rz_2 = rz * rz;
    T r_2  = (rx_2 + ry_2 + rz_2) * (1.0 / 3.0);

    T ml = addend[Cqi::mass] * 3;

    composite[Cqi::mass] += addend[Cqi::mass];
    composite[Cqi::qxx] += addend[Cqi::qxx] + ml * (rx_2 - r_2);
    composite[Cqi::qxy] += addend[Cqi::qxy] + ml * rx * ry;
    composite[Cqi::qxz] += addend[Cqi::qxz] + ml * rx * rz;
    composite[Cqi::qyy] += addend[Cqi::qyy] + ml * (ry_2 - r_2);
    composite[Cqi::qyz] += addend[Cqi::qyz] + ml * ry * rz;
    composite[Cqi::qzz] += addend[Cqi::qzz] + ml * (rz_2 - r_2);
}

/*! @brief Combine multipoles into a single multipole
 *
 * @tparam      T        float or double
 * @tparam      MType    Spherical multipole, quadrupole or octopole
 * @param[in]   begin    first index into @p sourceCenter and @p Multipole to aggregate
 * @param[in]   end      last index
 * @param[in]   Xout     the expansion (com) center of the output multipole
 * @param[in]   Xsrc     input multipole expansion (com) centers
 * @param[in]   Msrc     input multipoles
 * @param[out]  Mout     the aggregated output multipole
 */
template<class T, class MType, std::enable_if_t<MType{}.size() == CartesianQuadrupole<T>{}.size(), int> = 0>
void multipole2Multipole(int begin, int end, const Vec4<T>& Xout, const Vec4<T>* Xsrc, const MType* Msrc, MType& Mout)
{
    Mout = 0;
    for (int i = begin; i < end; i++)
    {
        const MType& Mi = Msrc[i];
        Vec4<T> Xi      = Xsrc[i];
        Vec3<T> dX      = makeVec3(Xout - Xi);
        addQuadrupole(Mout, dX, Mi);
    }
}

template<class MType, std::enable_if_t<IsCartesian<MType>{}, int> = 0>
HOST_DEVICE_FUN MType normalize(const MType& multipole)
{
    return multipole;
}

} // namespace ryoanji
