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
 * @brief Contains the object holding all stirring/turbulence related data
 *
 * @author Axel Sanz <axelsanzlechuga@gmail.com>
 */

#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "cstone/cuda/device_vector.h"
#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/primitives/accel_switch.hpp"

#include "sph/hydro_turb/create_modes.hpp"

namespace sph
{

template<class T, class Accelerator>
class TurbulenceData
{
    // resulting type is a thrust::device_vector if the Accelerator is a GPU, otherwise a std::vector
    using DeviceVector =
        typename cstone::AccelSwitchType<Accelerator, std::vector, cstone::DeviceVector>::template type<T>;

public:
    using RealType = T;

    TurbulenceData(const std::map<std::string, double>& constants, bool verbose)
        : solWeight(constants.at("solWeight"))
        , gen(size_t(constants.at("rngSeed")))
    {
        initModes(constants, verbose);
    }

    //! Number of dimensions
    size_t numDim{3};
    //! Variance of Ornstein-Uhlenbeck process
    T variance;
    T decayTime;
    //! Solenoidal weight
    T solWeight;
    //! Normalized solenoidal weight
    T solWeightNorm;

    uint64_t       numModes;
    std::vector<T> modes;
    std::vector<T> amplitudes;
    std::vector<T> phases;
    std::mt19937   gen;

    //! these are regenerated each step from phases above
    std::vector<T> phasesReal;
    std::vector<T> phasesImag;

    DeviceVector d_modes;
    DeviceVector d_amplitudes;
    DeviceVector d_phasesReal;
    DeviceVector d_phasesImag;

    //! @brief load or store all stateful turbulence data with file-based IO
    template<class Archive>
    void loadOrStore(Archive* ar)
    {
        std::string prefix = "turbulence::";
        ar->stepAttribute(prefix + "variance", &variance, 1);
        ar->stepAttribute(prefix + "decayTime", &decayTime, 1);
        ar->stepAttribute(prefix + "solWeight", &solWeight, 1);
        ar->stepAttribute(prefix + "solWeightNorm", &solWeightNorm, 1);
        ar->stepAttribute(prefix + "numModes", &numModes, 1);

        resize(numModes);

        ar->stepAttribute(prefix + "modes", modes.data(), modes.size());
        ar->stepAttribute(prefix + "amplitudes", amplitudes.data(), amplitudes.size());
        ar->stepAttribute(prefix + "phases", phases.data(), phases.size());

        uploadModes();

        std::stringstream s;
        s << gen;
        std::string engineState = s.str();

        int64_t rngStateSize = ar->stepAttributeSize("rngEngineState");
        rngStateSize         = (rngStateSize) ? rngStateSize : int64_t(engineState.size());

        engineState.resize(rngStateSize);
        ar->stepAttribute("rngEngineState", engineState.data(), rngStateSize);

        s = std::stringstream{};
        s << engineState;
        s >> gen;
    }

private:
    void resize(size_t numModes_)
    {
        modes.resize(numDim * numModes_);
        phases.resize(2 * numDim * numModes_);
        amplitudes.resize(numModes_);

        phasesReal.resize(numDim * numModes_);
        phasesImag.resize(numDim * numModes_);
    }

    void uploadModes()
    {
        if constexpr (cstone::HaveGpu<Accelerator>{})
        {
            // upload data to the GPU
            d_modes      = modes;
            d_amplitudes = amplitudes;
        }
    }

    /*! @brief initialize parameters, modes and amplitudes which stay constant during the simulation
     *
     * @param constants  settings as key-value pairs
     * @param verbose    whether to print some diagnostics
     *
     * Also fills the phases with a random gaussian distribution, which  will be overwritten when loading from file.
     */
    void initModes(const std::map<std::string, double>& constants, bool verbose)
    {
        double eps         = constants.at("epsilon");
        size_t stMaxModes  = size_t(constants.at("stMaxModes"));
        double Lbox        = constants.at("Lbox");
        double velocity    = constants.at("stMachVelocity");
        size_t stSpectForm = size_t(constants.at("stSpectForm"));
        double powerLawExp = constants.at("powerLawExp");
        double anglesExp   = constants.at("anglesExp");

        double twopi   = 2.0 * M_PI;
        double energy  = constants.at("stEnergyPrefac") * std::pow(velocity, 3) / Lbox;
        double stirMin = (1.0 - eps) * twopi / Lbox;
        double stirMax = (3.0 + eps) * twopi / Lbox;

        decayTime = Lbox / (2.0 * velocity);
        variance  = std::sqrt(energy / decayTime);
        // this makes the rms force const irrespective of the solenoidal weight
        solWeightNorm = std::sqrt(3.0) * std::sqrt(3.0 / T(numDim)) /
                        std::sqrt(1.0 - 2.0 * solWeight + T(numDim) * solWeight * solWeight);

        amplitudes.resize(stMaxModes);
        modes.resize(stMaxModes * numDim);

        createStirringModes(*this, Lbox, Lbox, Lbox, stMaxModes, stirMax, stirMin, numDim, stSpectForm, powerLawExp,
                            anglesExp, verbose);

        resize(numModes);
        uploadModes();

        // fill phases with normal gaussian distributed random values with mean 0 and std-dev "variance"
        std::normal_distribution<T> dist(0, variance);
        std::generate(phases.begin(), phases.end(), [this, &dist]() { return dist(gen); });
    }
};

} // namespace sph
