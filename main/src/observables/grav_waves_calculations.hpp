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

/*! @file calculations for the gravitational waves observable
 *          References: Centrella & McMillan ApJ, 416 (1993), R. M. Cabezon Phd Thesis (2010)
 *
 *  @author Lukas Schmidt
 */

#include <cmath>

namespace sphexa
{
struct QIdx
{
    enum IndexLabels
    {
        xx = 0,
        yy = 1,
        zz = 2,
        xy = 3,
        xz = 4,
        yz = 5
    };
};

//! @brief calculates the two polarization modes of the gravitational waves for a given quadrupole momentum
template<class T>
void computeHtt(std::array<T, 6> quadpoleMomentum, T theta, T phi, T* httplus, T* httcross)
{
    T g       = 6.6726e-8;                          // Gravitational constant
    T c       = 2.997924562e10;                     // Speed of light
    T gwunits = g / std::pow(c, 4) / 3.08568025e22; // Gravitational Waves unit at 10kpc

    T sin2t  = std::sin(2.0 * theta);
    T sin2p  = std::sin(2.0 * phi);
    T cos2p  = std::cos(2.0 * phi);
    T sint   = std::sin(theta);
    T sinp   = std::sin(phi);
    T cost   = std::cos(theta);
    T cosp   = std::cos(phi);
    T sqsint = sint * sint;
    T sqsinp = sinp * sinp;
    T sqcost = cost * cost;
    T sqcosp = cosp * cosp;

    T dot2ibartt = (quadpoleMomentum[QIdx::xx] * sqcosp + quadpoleMomentum[QIdx::yy] * sqsinp +
                    quadpoleMomentum[QIdx::xy] * sin2p) *
                       sqcost +
                   quadpoleMomentum[QIdx::zz] * sqsint -
                   (quadpoleMomentum[QIdx::xz] * cosp + quadpoleMomentum[QIdx::yz] * sinp) * sin2t;

    T dot2ibarpp =
        quadpoleMomentum[QIdx::xx] * sqsinp + quadpoleMomentum[QIdx::yy] * sqcosp - quadpoleMomentum[QIdx::xy] * sin2p;

    T dot2ibartp = 0.5 * (quadpoleMomentum[QIdx::yy] - quadpoleMomentum[QIdx::xx]) * cost * sin2p +
                   quadpoleMomentum[QIdx::xy] * cost * cos2p +
                   (quadpoleMomentum[QIdx::xz] * sinp - quadpoleMomentum[QIdx::yz] * cosp) * sint;

    *httplus  = (dot2ibartt - dot2ibarpp) * gwunits;
    *httcross = 2.0 * dot2ibartp * gwunits;
}

//!@brief calculates the second derivative of the quadrupole momentum
template<class Tc, class Tv, class Ta, class Tm>
Tc d2QuadpoleMomentum(size_t begin, size_t end, int dim1, int dim2, const Tc* x, const Tc* y, const Tc* z, const Tv* vx,
                      const Tv* vy, const Tv* vz, const Ta* ax, const Ta* ay, const Ta* az, const Tm* m)
{
    Tc out = 0.0;

    std::array<const Tc*, 3> coords = {x, y, z};
    std::array<const Tv*, 3> vel    = {vx, vy, vz};
    std::array<const Ta*, 3> acc    = {ax, ay, az};

    if (dim1 == dim2)
    {
#pragma omp parallel for reduction(+ : out)
        for (size_t i = begin; i < end; i++)
        {
            Tv scalv2        = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];
            Tc coordDotAccel = x[i] * ax[i] + y[i] * ay[i] + z[i] * az[i];

            out +=
                (3.0 * (vel[dim1][i] * vel[dim1][i] + coords[dim1][i] * acc[dim1][i]) - scalv2 - coordDotAccel) * m[i];
        }
        return out * 2.0 / 3.0;
    }
    else
    {
#pragma omp parallel for reduction(+ : out)
        for (size_t i = begin; i < end; i++)
        {
            out +=
                (2.0 * vel[dim1][i] * vel[dim2][i] + acc[dim1][i] * coords[dim2][i] + coords[dim1][i] * acc[dim2][i]) *
                m[i];
        }
        return out;
    }
}

} // namespace sphexa