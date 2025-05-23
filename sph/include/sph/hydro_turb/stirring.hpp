/*
 * MIT License
 *
 * Copyright (c) 2022 Politechnical University of Catalonia UPC
 *               2022 University of Basel
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
 * @brief  Implementation of stirring accelerations
 * @author Axel Sanz <axel.sanz@estudiantat.upc.edu>
 */

#pragma once

#include <cmath>

#include "cstone/cuda/annotation.hpp"
#include "cstone/traversal/groups.hpp"
#include "cstone/util/tuple.hpp"

namespace sph
{

//! @brief compute stirring acceleration for a single particle
template<class Tc, class Ta, class T>
HOST_DEVICE_FUN auto stirParticle(size_t ndim, Tc xi, Tc yi, Tc zi, size_t numModes, const T* modes, const T* phaseReal,
                                  const T* phaseImag, const T* amplitudes)
{
    Ta turbAx = 0.0;
    Ta turbAy = 0.0;
    Ta turbAz = 0.0;

    for (size_t m = 0; m < numModes; ++m)
    {
        size_t m_ndim = m * ndim;

        T cosxk_im = std::cos(modes[m_ndim + 2] * zi);
        T sinxk_im = std::sin(modes[m_ndim + 2] * zi);

        T cosxj_im = std::cos(modes[m_ndim + 1] * yi);
        T sinxj_im = std::sin(modes[m_ndim + 1] * yi);

        T cosxi_im = std::cos(modes[m_ndim] * xi);
        T sinxi_im = std::sin(modes[m_ndim] * xi);

        //  these are the real and imaginary parts, respectively, of
        //     e^{ i \vec{k} \cdot \vec{x} }
        //          = cos(kx*x + ky*y + kz*z) + i sin(kx*x + ky*y + kz*z)
        T realtrigterms = (cosxi_im * cosxj_im - sinxi_im * sinxj_im) * cosxk_im -
                          (sinxi_im * cosxj_im + cosxi_im * sinxj_im) * sinxk_im;

        T imtrigterms = cosxi_im * (cosxj_im * sinxk_im + sinxj_im * cosxk_im) +
                        sinxi_im * (cosxj_im * cosxk_im - sinxj_im * sinxk_im);

        turbAx += amplitudes[m] * (phaseReal[m_ndim] * realtrigterms - phaseImag[m_ndim] * imtrigterms);
        turbAy += amplitudes[m] * (phaseReal[m_ndim + 1] * realtrigterms - phaseImag[m_ndim + 1] * imtrigterms);
        turbAz += amplitudes[m] * (phaseReal[m_ndim + 2] * realtrigterms - phaseImag[m_ndim + 2] * imtrigterms);
    }

    return util::tuple<Ta, Ta, Ta>(turbAx, turbAy, turbAz);
}

/*! @brief Adds the stirring accelerations to the provided particle accelerations
 *
 * @tparam        Tc                float or double
 * @tparam        Ta                float or double
 * @tparam        T                 float or double
 * @param[in]     startIndex        first index of particles
 * @param[in]     endIndex          last index of particles
 * @param[in]     numDim            number of dimensions
 * @param[in]     x                 x components of particle positions
 * @param[in]     y                 y components of particle positions
 * @param[in]     z                 z components of particle positionSs
 * @param[inout]  ax                x component of accelerations
 * @param[inout]  ay                y component of accelerations
 * @param[inout]  az                z component of accelerations
 * @param[in]     numModes          number of modes
 * @param[in]     modes             matrix (st_nmodes x dimension) containing modes
 * @param[in]     phaseReal         matrix (st_nmodes x dimension) containing real phases
 * @param[in]     phaseImag         matrix (st_nmodes x dimension) containing imaginary phases
 * @param[in]     amplitudes        amplitudes of modes
 * @param[in]     solWeightNorm     normalized solenoidal weight
 *
 */
template<class Tc, class Ta, class T>
void computeStirring(cstone::GroupView grp, size_t numDim, const Tc* x, const Tc* y, const Tc* z, Ta* ax, Ta* ay,
                     Ta* az, size_t numModes, const T* modes, const T* phaseReal, const T* phaseImag,
                     const T* amplitudes, T solWeightNorm)
{
#pragma omp parallel for schedule(static)
    for (cstone::LocalIndex gi = 0; gi < grp.numGroups; ++gi)
    {
        for (cstone::LocalIndex i = grp.groupStart[gi]; i < grp.groupEnd[gi]; ++i)
        {
            auto [turbAx, turbAy, turbAz] =
                stirParticle<Tc, Ta, T>(numDim, x[i], y[i], z[i], numModes, modes, phaseReal, phaseImag, amplitudes);

            ax[i] += solWeightNorm * turbAx;
            ay[i] += solWeightNorm * turbAy;
            az[i] += solWeightNorm * turbAz;
        }
    }
}

template<class Tc, class Ta, class T>
extern void computeStirringGpu(cstone::GroupView grp, size_t numDim, const Tc* x, const Tc* y, const Tc* z, Ta* ax,
                               Ta* ay, Ta* az, size_t numModes, const T* modes, const T* phaseReal, const T* phaseImag,
                               const T* amplitudes, T solWeightNorm);

} // namespace sph
