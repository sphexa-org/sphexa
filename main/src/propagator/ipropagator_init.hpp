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
 * @brief Propagator initialization
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author ChristopherBignamini <christopher.bignamini@gmail.com>
 */

#pragma once

#include <memory>

#include "propagator.hpp"
#include "init/settings.hpp"
#include "sphexa/simulation_data.hpp"


namespace sphexa
{

template<class DomainType, class ParticleDataType>
struct PropLib
{

    using PropPtr = std::unique_ptr<Propagator<DomainType, ParticleDataType>>;

    static PropPtr makeHydroVeProp(std::ostream& output, size_t rank, bool avClean);
    static PropPtr makeHydroProp(std::ostream& output, size_t rank);
    static PropPtr makeHydroVeBdtProp(std::ostream& output, size_t rank, const InitSettings& settings, bool avClean);
#ifdef SPH_EXA_HAVE_GRACKLE
    static PropPtr makeHydroGrackleProp(std::ostream& output, size_t rank, const InitSettings& settings);
#endif
    static PropPtr makeNbodyProp(std::ostream& output, size_t rank);
    static PropPtr makeTurbVeBdtProp(std::ostream& output, size_t rank, const InitSettings& settings, bool avClean);
    static PropPtr makeTurbVeProp(std::ostream& output, size_t rank, const InitSettings& settings, bool avClean);
};

}
