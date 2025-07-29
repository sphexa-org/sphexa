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
 * @brief Turbulence driver calling all stirring subroutines
 *
 * @author Axel Sanz <axelsanzlechuga@gmail.com>
 */

#pragma once

#include <random>

#include "cstone/cuda/cuda_utils.hpp"
#include "sph/hydro_turb/turbulence_data.hpp"
#include "sph/hydro_turb/stirring.hpp"
#include "sph/hydro_turb/phases.hpp"

namespace sph
{

/*! @brief Generates an Ornstein-Uhlenbeck sequence.
 *
 *   @param[inout] phases   the Ornstein-Uhlenbeck phases to be updated
 *   @param[in]    stddev   standard deviation of the distribution
 *   @param[in]    dt       timestep
 *   @param[in]    ts       auto-correlation time
 *
 * The sequence x_n is a Markov process that takes the previous value,
 *   weights by an exponential damping factor with a given correlation
 *   time "ts", and drives by adding a Gaussian random variable with
 *   variance "variance", weighted by a second damping factor, also
 *   with correlation time "ts". For a timestep of dt, this sequence
 *   can be written as :
 *
 *     x_n+1 = f x_n + sigma * sqrt (1 - f**2) z_n
 *
 * where f = exp (-dt / ts), z_n is a Gaussian random variable drawn
 * from a Gaussian distribution with unit variance, and sigma is the
 * desired variance of the OU sequence. (See Bartosch, 2001).
 *
 * The resulting sequence should satisfy the properties of zero mean,
 *   and stationary (independent of portion of sequence) RMS equal to
 *   "variance". Its power spectrum in the time domain can vary from
 *   white noise to "brown" noise (P (f) = const. to 1 / f^2).
 *
 * References :
 *    Bartosch, 2001
 * http://octopus.th.physik.uni-frankfurt.de/~bartosch/publications/IntJMP01.pdf
 *   Finch, 2004
 * http://pauillac.inria.fr/algo/csolve/ou.pdf
 *         Uhlenbeck & Ornstein, 1930
 * http://prola.aps.org/abstract/PR/v36/i5/p823_1
 *
 * Eswaran & Pope 1988
 */
template<class T>
void updateNoise(std::vector<T>& phases, T stddev, T dt, T ts, std::mt19937& gen)
{
    T dampingA = std::exp(-dt / ts);
    T dampingB = std::sqrt(1.0 - dampingA * dampingA);
    // mean 0 and unit standard deviation
    std::normal_distribution<T> dist(0, 1);

    for (size_t i = 0; i < phases.size(); i++)
    {
        T randomNumber = dist(gen);
        phases[i]      = phases[i] * dampingA + stddev * dampingB * randomNumber;
    }
}

/*! @brief Adds the stirring motion to particle accelerations
 *
 * @tparam Dataset          SPH hydro dataset
 * @param  startIndex       first locally owned particle index
 * @param  endIndex         last locally owned particle index
 * @param  d                Hydro data
 * @param  turb             Turbulence modes
 */
template<class Dataset>
void driveTurbulence(GroupView grp, Dataset& d,
                     TurbulenceData<typename Dataset::RealType, typename Dataset::AcceleratorType>& turb)
{
    updateNoise(turb.phases, turb.variance, d.minDt, turb.decayTime, turb.gen);
    computePhases(turb.numModes, turb.numDim, turb.phases, turb.solWeight, turb.modes, turb.phasesReal,
                  turb.phasesImag);

    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        // upload mode data to the device
        turb.d_phasesReal = turb.phasesReal;
        turb.d_phasesImag = turb.phasesImag;

        computeStirringGpu(grp, turb.numDim, rawPtr(d.devData.x), rawPtr(d.devData.y), rawPtr(d.devData.z),
                           rawPtr(d.devData.ax), rawPtr(d.devData.ay), rawPtr(d.devData.az), turb.numModes,
                           rawPtr(turb.d_modes), rawPtr(turb.d_phasesReal), rawPtr(turb.d_phasesImag),
                           rawPtr(turb.d_amplitudes), turb.solWeightNorm);
        syncGpu();
    }
    else
    {
        computeStirring(grp, turb.numDim, d.x.data(), d.y.data(), d.z.data(), d.ax.data(), d.ay.data(), d.az.data(),
                        turb.numModes, turb.modes.data(), turb.phasesReal.data(), turb.phasesImag.data(),
                        turb.amplitudes.data(), turb.solWeightNorm);
    }
}

} // namespace sph
