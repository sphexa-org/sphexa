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
#include "kernel.hpp"

namespace ryoanji
{

template<class T>
using CartesianQuadrupole = util::array<T, 8>; // Use 8 here for memory alignment, even though we only use 7 elements.

template<class MType>
struct IsCartesian : public stl::integral_constant<size_t, MType{}.size() == CartesianQuadrupole<float>{}.size()>
{
};

//! @brief CartesianQuadrupole index names
struct Cqi
{
    enum IndexNames
    {
        mass  = 0,
        qxx   = 1,
        qxy   = 2,
        qxz   = 3,
        qyy   = 4,
        qyz   = 5,
        qzz   = 6,
        trace = 7,
    };
};

/*! @brief Compute the monopole and quadruple moments from particle coordinates
 *
 * @tparam      T1     float or double
 * @tparam      T2     float or double
 * @tparam      T3     float or double
 * @param[in]   x      x coordinate array
 * @param[in]   y      y coordinate array
 * @param[in]   z      z coordinate array
 * @param[in]   m      masses array
 * @param[in]   begin  first particle to access in coordinate arrays
 * @param[in]   end    last particle to access in coordinate arrays
 * @param[in]   center center of mass of particles in input range
 * @param[out]  gv     output quadrupole
 */
template<class T1, class T2, class T3>
HOST_DEVICE_FUN void P2M(const T1* x, const T1* y, const T1* z, const T2* m, LocalIndex begin, LocalIndex end,
                         const Vec4<T1>& center, CartesianQuadrupole<T3>& gv)
{
    gv = T3(0);
    if (begin == end) { return; }

    for (LocalIndex i = begin; i < end; ++i)
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

    T3 traceQ = gv[Cqi::qxx] + gv[Cqi::qyy] + gv[Cqi::qzz];

    gv[Cqi::trace] = traceQ;

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
    T       rx = dX[0];
    T       ry = dX[1];
    T       rz = dX[2];

    gv[Cqi::qxx] = gv.qxx - rx * rx * gv[Cqi::mass];
    gv[Cqi::qxy] = gv.qxy - rx * ry * gv[Cqi::mass];
    gv[Cqi::qxz] = gv.qxz - rx * rz * gv[Cqi::mass];
    gv[Cqi::qyy] = gv.qyy - ry * ry * gv[Cqi::mass];
    gv[Cqi::qyz] = gv.qyz - ry * rz * gv[Cqi::mass];
    gv[Cqi::qzz] = gv.qzz - rz * rz * gv[Cqi::mass];

    T traceQ = gv[Cqi::qxx] + gv[Cqi::qyy] + gv[Cqi::qzz];

    gv[Cqi::trace] = traceQ;

    // remove trace
    gv[Cqi::qxx] = 3 * gv[Cqi::qxx] - traceQ;
    gv[Cqi::qyy] = 3 * gv[Cqi::qyy] - traceQ;
    gv[Cqi::qzz] = 3 * gv[Cqi::qzz] - traceQ;
    gv[Cqi::qxy] *= 3;
    gv[Cqi::qxz] *= 3;
    gv[Cqi::qyz] *= 3;
}

/*! @brief apply gravitational interaction with a multipole to a particle
 *
 * @tparam        T1         float or double
 * @tparam        T2         float or double
 * @param[in]     tx         target particle x coordinate
 * @param[in]     ty         target particle y coordinate
 * @param[in]     tz         target particle z coordinate
 * @param[in]     center     source center of mass
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
template<class Ta, class Tc, class Tmp>
HOST_DEVICE_FUN DEVICE_INLINE Vec4<Ta> M2P(Vec4<Ta> acc, const Vec3<Tc>& target, const Vec3<Tc>& center,
                                           const CartesianQuadrupole<Tmp>& multipole)
{
    auto r = target - center;

    auto r2       = norm2(r);
    auto r_minus1 = inverseSquareRoot(r2);
    auto r_minus2 = r_minus1 * r_minus1;
    auto r_minus5 = r_minus2 * r_minus2 * r_minus1;

    auto Qrx = r[0] * multipole[Cqi::qxx] + r[1] * multipole[Cqi::qxy] + r[2] * multipole[Cqi::qxz];
    auto Qry = r[0] * multipole[Cqi::qxy] + r[1] * multipole[Cqi::qyy] + r[2] * multipole[Cqi::qyz];
    auto Qrz = r[0] * multipole[Cqi::qxz] + r[1] * multipole[Cqi::qyz] + r[2] * multipole[Cqi::qzz];

    auto rQr = r[0] * Qrx + r[1] * Qry + r[2] * Qrz;
    //                  rQr quad-term           mono-term
    //                      |                     |
    auto rQrAndMonopole = (Ta(-2.5) * rQr * r_minus5 - multipole[Cqi::mass] * r_minus1) * r_minus2;

    //       Qr Quad-term
    return acc + Vec4<Ta>{-(multipole[Cqi::mass] * r_minus1 + Ta(0.5) * r_minus5 * rQr),
                          r_minus5 * Qrx + rQrAndMonopole * r[0], r_minus5 * Qry + rQrAndMonopole * r[1],
                          r_minus5 * Qrz + rQrAndMonopole * r[2]};
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
template<class T, class Tc>
HOST_DEVICE_FUN void addQuadrupole(CartesianQuadrupole<T>& composite, Vec3<Tc> dX, const CartesianQuadrupole<T>& addend)
{
    Tc rx = dX[0];
    Tc ry = dX[1];
    Tc rz = dX[2];

    Tc rx_2 = rx * rx;
    Tc ry_2 = ry * ry;
    Tc rz_2 = rz * rz;
    Tc r_2  = (rx_2 + ry_2 + rz_2) * (1.0 / 3.0);

    Tc ml = addend[Cqi::mass] * 3;

    composite[Cqi::trace] = composite[Cqi::trace] + addend[Cqi::trace] + ml * r_2;

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
template<class T, class MType, std::enable_if_t<IsCartesian<MType>{}, int> = 0>
HOST_DEVICE_FUN void M2M(int begin, int end, const Vec4<T>& Xout, const Vec4<T>* Xsrc, const MType* Msrc, MType& Mout)
{
    Mout = 0;
    for (int i = begin; i < end; i++)
    {
        const MType& Mi = Msrc[i];
        Vec4<T>      Xi = Xsrc[i];
        Vec3<T>      dX = makeVec3(Xout - Xi);
        addQuadrupole(Mout, dX, Mi);
    }
}

} // namespace ryoanji
