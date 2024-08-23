/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *               2024 University of Basel
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
 * @brief Translation unit for the propagator initializer library
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author ChristopherBignamini <christopher.bignamini@gmail.com>
 */


#include "ipropagator_init.hpp"

#include "ipropagator.hpp"
#include "nbody.hpp"
#include "std_hydro.hpp"
#include "ve_hydro.hpp"
#include "ve_hydro_bdt.hpp"
#ifdef SPH_EXA_HAVE_GRACKLE
#include "std_hydro_grackle.hpp"
#endif
#include "turb_ve.hpp"


namespace sphexa
{

template<class DomainType, class ParticleDataType>
std::unique_ptr<Propagator<DomainType, ParticleDataType>>
PropInitializers<DomainType, ParticleDataType>::makeHydroVeProp(std::ostream& output, size_t rank, bool avClean)
{
    if (avClean) { return std::make_unique<HydroVeProp<true, DomainType, ParticleDataType>>(output, rank); }
    else { return std::make_unique<HydroVeProp<false, DomainType, ParticleDataType>>(output, rank); }
}

template<class DomainType, class ParticleDataType>
std::unique_ptr<Propagator<DomainType, ParticleDataType>>
PropInitializers<DomainType, ParticleDataType>::makeHydroVeBdtProp(std::ostream& output, size_t rank, const InitSettings& settings, bool avClean)
{
    if (avClean) { return std::make_unique<HydroVeBdtProp<true, DomainType, ParticleDataType>>(output, rank, settings); }
    else { return std::make_unique<HydroVeBdtProp<false, DomainType, ParticleDataType>>(output, rank, settings); }
}

template<class DomainType, class ParticleDataType>
std::unique_ptr<Propagator<DomainType, ParticleDataType>>
PropInitializers<DomainType, ParticleDataType>::makeHydroProp(std::ostream& output, size_t rank)
{
    return std::make_unique<HydroProp<DomainType, ParticleDataType>>(output, rank);
}

#ifdef SPH_EXA_HAVE_GRACKLE
template<class DomainType, class ParticleDataType>
std::unique_ptr<Propagator<DomainType, ParticleDataType>>
PropInitializers<DomainType, ParticleDataType>::makeHydroGrackleProp(std::ostream& output, size_t rank, const InitSettings& settings)
{
    return std::make_unique<HydroGrackleProp<DomainType, ParticleDataType>>(output, rank, settings);
}
#endif

template<class DomainType, class ParticleDataType>
std::unique_ptr<Propagator<DomainType, ParticleDataType>>
PropInitializers<DomainType, ParticleDataType>::makeNbodyProp(std::ostream& output, size_t rank)
{
    return std::make_unique<NbodyProp<DomainType, ParticleDataType>>(output, rank);
}

template<class DomainType, class ParticleDataType>
std::unique_ptr<Propagator<DomainType, ParticleDataType>>
PropInitializers<DomainType, ParticleDataType>::makeTurbVeBdtProp(std::ostream& output, size_t rank, const InitSettings& settings, bool avClean)
{
    if (avClean) { return std::make_unique<TurbVeBdtProp<true, DomainType, ParticleDataType>>(output, rank, settings); }
    else { return std::make_unique<TurbVeBdtProp<false, DomainType, ParticleDataType>>(output, rank, settings); }
}

template<class DomainType, class ParticleDataType>
std::unique_ptr<Propagator<DomainType, ParticleDataType>>
PropInitializers<DomainType, ParticleDataType>::makeTurbVeProp(std::ostream& output, size_t rank, const InitSettings& settings, bool avClean)
{
    if (avClean) { return std::make_unique<TurbVeProp<true, DomainType, ParticleDataType>>(output, rank, settings); }
    else { return std::make_unique<TurbVeProp<false, DomainType, ParticleDataType>>(output, rank, settings); }
}

#ifdef USE_CUDA
template struct PropInitializers<cstone::Domain<unsigned long, double, cstone::GpuTag>, SimulationData<cstone::GpuTag> >;
#else
template struct PropInitializers<cstone::Domain<unsigned long, double, cstone::CpuTag>, SimulationData<cstone::CpuTag> >;
#endif

/* #ifdef USE_CUDA
template struct PropInitializers<class DomainType, sphexa::SimulationData<cstone::GpuTag>>;
#else
template struct PropInitializers<class DomainType, sphexa::SimulationData<cstone::CpuTag>>;
#endif
 */
} // namespace sphexa

