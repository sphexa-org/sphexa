
#include <iostream>
#include <memory>

#include "cstone/primitives/accel_switch.hpp"

#include "gravitational_waves.hpp"
#include "time_energies.hpp"
#include "time_energy_growth.hpp"
#include "turbulence_mach_rms.hpp"
#include "wind_bubble_fraction.hpp"

#include "iobservables.hpp"

namespace sphexa
{

template<class Dataset>
std::unique_ptr<IObservables<Dataset>> Observables<Dataset>::makeGravWaveObs(std::ostream& out, double theta,
                                                                             double phi)
{
    return std::make_unique<GravWaves<Dataset>>(out, theta, phi);
}

template<class Dataset>
std::unique_ptr<IObservables<Dataset>> Observables<Dataset>::makeTimeEnergyObs(std::ostream& out)
{
    return std::make_unique<TimeAndEnergy<Dataset>>(out);
}

template<class Dataset>
std::unique_ptr<IObservables<Dataset>> Observables<Dataset>::makeTimeEnergyGrowthObs(std::ostream& out)
{
    return std::make_unique<TimeEnergyGrowth<Dataset>>(out);
}

template<class Dataset>
std::unique_ptr<IObservables<Dataset>> Observables<Dataset>::makeTurbMachObs(std::ostream& out)
{
    return std::make_unique<TurbulenceMachRMS<Dataset>>(out);
}

template<class Dataset>
std::unique_ptr<IObservables<Dataset>> Observables<Dataset>::makeWindBubbleObs(std::ostream& out, double rhoI,
                                                                               double uExt, double r)
{
    return std::make_unique<WindBubble<Dataset>>(out, rhoI, uExt, r);
}

#ifdef USE_CUDA
template struct Observables<SimulationData<cstone::GpuTag>>;
#else
template struct Observables<SimulationData<cstone::CpuTag>>;
#endif

} // namespace sphexa
