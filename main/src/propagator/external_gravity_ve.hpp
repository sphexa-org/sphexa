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
 * @brief A Propagator class for modern SPH with generalized volume elements
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "sph/sph.hpp"
#include "ve_hydro.hpp"

namespace sphexa
{

using namespace sph;
using cstone::FieldStates;

//! @brief VE hydro propagator with artificial gravity for the Rayleigh-Taylor test case
template<bool avClean, class DomainType, class DataType>
class ExternalGravityVeProp final : public HydroVeProp<avClean, DomainType, DataType>
{
    using Base = HydroVeProp<avClean, DomainType, DataType>;
    using Base::rank_;
    using Base::timer;

    using RealType = typename DataType::RealType;
    RealType gravityConstant;

public:
    ExternalGravityVeProp(std::ostream& output, size_t rank, RealType grav)
        : Base(output, rank)
        , gravityConstant(grav)
    {
    }

    void activateFields(DataType& simData) override
    {
        Base::activateFields(simData);
        simData.hydro.setDependent("markRamp");
        simData.hydro.devData.setDependent("markRamp");
    }

    void computeForces(DomainType& domain, DataType& simData) override
    {
        Base::computeForces(domain, simData);

        auto&  d     = simData.hydro;
        size_t first = domain.startIndex();
        size_t last  = domain.endIndex();

        computeMarkRamp(Base::groups_.view(), d, domain.box());
        timer.step("MarkRamp");

        // fixedBoundaryForceCorrection(first, last, d, domain.box());
        artificialGravity(first, last, d, gravityConstant);
        timer.step("ArtificialGravity");
    }
};

} // namespace sphexa
