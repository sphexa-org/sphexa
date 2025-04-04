/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich, University of Zurich, University of Basel
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
 * @brief Unit tests for ParticlesData
 *
 * @author Noah Kubli <noah.kubli@uzh.ch>
 */

#pragma once

#include "evrard_init.hpp"

#include "cooling/cooler.hpp"
#include "cooling/init_chemistry.h"

namespace sphexa
{

InitSettings evrardCoolingConstants()
{
    return {{"gravConstant", 1.},
            {"r", 1.},
            {"mTotal", 1.},
            {"gamma", 5. / 3.},
            {"u0", 0.05},
            {"minDt", 1e-4},
            {"minDt_m1", 1e-4},
            {"mui", 10},
            {"ng0", 100},
            {"ngmax", 150},
            {"cooling::use_grackle", 1},
            {"cooling::with_radiative_cooling", 1},
            {"cooling::primordial_chemistry", 1},
            {"cooling::dust_chemistry", 0},
            {"cooling::metal_cooling", 0},
            {"cooling::UVbackground", 0},
            {"cooling::m_code_in_ms", 1e16},
            {"cooling::l_code_in_kpc", 46400.}};

}

template<class Dataset>
class EvrardGlassSphereCooling : public EvrardGlassSphere<Dataset>
{
    using Base = EvrardGlassSphere<Dataset>;
    mutable InitSettings settings_;

public:
    EvrardGlassSphereCooling(std::string initBlock, std::string settingsFile, IFileReader* reader)
        : EvrardGlassSphere<Dataset>(initBlock, settingsFile, reader)
    {
        Dataset d;
        settings_ = buildSettings(d, evrardCoolingConstants(), settingsFile, reader);
        Base::resetConstants(settings_);
    }

    cstone::Box<typename Dataset::RealType> init(int rank, int numRanks, size_t cbrtNumPart, Dataset& simData,
                                                 IFileReader* reader) const override
    {
        auto box = Base::init(rank, numRanks, cbrtNumPart, simData, reader);
        std::fill(simData.hydro.u.begin(), simData.hydro.u.end(), std::get<ScalarValue>(settings_.at("u0").getValue()));
        cooling::initChemistryData(simData.chem, simData.hydro.x.size());

        return box;
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};

} // namespace sphexa