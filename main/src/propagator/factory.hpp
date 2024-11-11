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
 * @brief Evaluate choice of propagator
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Jose A. Escartin <ja.escartin@gmail.com>
 */

#pragma once

#include <variant>

#include "ipropagator.hpp"
#include "nbody.hpp"
#include "std_hydro.hpp"
#include "ve_hydro.hpp"
#include "ve_hydro_bdt.hpp"
#ifdef SPH_EXA_HAVE_GRACKLE
#include "std_hydro_grackle.hpp"
#endif
#include "turb_ve.hpp"
#include "std_planet.hpp"

namespace sphexa
{

template<class DomainType, class ParticleDataType>
std::unique_ptr<Propagator<DomainType, ParticleDataType>>
propagatorFactory(const std::string& choice, bool avClean, std::ostream& output, size_t rank, const InitSettings& s)
{
    if (choice == "ve")
    {
        if (avClean) { return std::make_unique<HydroVeProp<true, DomainType, ParticleDataType>>(output, rank); }
        else { return std::make_unique<HydroVeProp<false, DomainType, ParticleDataType>>(output, rank); }
    }
    if (choice == "ve-bdt")
    {
        if (avClean) { return std::make_unique<HydroVeBdtProp<true, DomainType, ParticleDataType>>(output, rank, s); }
        else { return std::make_unique<HydroVeBdtProp<false, DomainType, ParticleDataType>>(output, rank, s); }
    }
    if (choice == "std") { return std::make_unique<HydroProp<DomainType, ParticleDataType>>(output, rank); }
#ifdef SPH_EXA_HAVE_GRACKLE
    if (choice == "std-cooling")
    {
        return std::make_unique<HydroGrackleProp<DomainType, ParticleDataType>>(output, rank, s);
    }
#endif
    if (choice == "nbody") { return std::make_unique<NbodyProp<DomainType, ParticleDataType>>(output, rank); }
    if (choice == "turbulence")
    {
        if (avClean) { return std::make_unique<TurbVeBdtProp<true, DomainType, ParticleDataType>>(output, rank, s); }
        else { return std::make_unique<TurbVeBdtProp<false, DomainType, ParticleDataType>>(output, rank, s); }
    }
    if (choice == "turbulence-ve")
    {
        if (avClean) { return std::make_unique<TurbVeProp<true, DomainType, ParticleDataType>>(output, rank, s); }
        else { return std::make_unique<TurbVeProp<false, DomainType, ParticleDataType>>(output, rank, s); }
    }
    if (choice == "std-planet") { return std::make_unique<PlanetProp<DomainType, ParticleDataType>>(output, rank, s); }

    throw std::runtime_error("Unknown propagator choice: " + choice);
}

} // namespace sphexa
