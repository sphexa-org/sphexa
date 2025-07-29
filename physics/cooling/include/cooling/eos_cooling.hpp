//
// Created by Noah Kubli on 09.07.23.
//

#pragma once

namespace cooling
{

//! @brief the maximum time-step based on local particles that Grackle can tolerate
template<class Dataset, typename Cooler, typename Chem>
auto coolingTimestep(size_t first, size_t last, Dataset& d, Cooler& cooler, Chem& chem)
{
    using T               = typename Dataset::RealType;
    using CoolingFields   = typename Cooler::CoolingFields;
    const auto* rho       = d.rho.data();
    const auto* u         = d.u.data();
    const auto  chemistry = cstone::getPointers(get<CoolingFields>(chem), 0);

    T minCt = cooler.cooling_timestep(rho, u, chemistry, first, last);

    return minCt;
}

template<typename HydroData, typename ChemData, typename Cooler>
void eos_cooling(size_t startIndex, size_t endIndex, HydroData& d, ChemData& chem, Cooler& cooler)
{
    using CoolingFields   = typename Cooler::CoolingFields;
    using T               = typename HydroData::RealType;
    const auto* rho       = d.rho.data();
    const auto* u         = d.u.data();
    const auto  chemistry = cstone::getPointers(get<CoolingFields>(chem), 0);

    auto* p = d.p.data();
    auto* c = d.c.data();

    cooler.computePressures(rho, u, chemistry, p, startIndex, endIndex);

    // Write adiabatic indices into c (sound speed) first
    cooler.computeAdiabaticIndices(rho, u, chemistry, c, startIndex, endIndex);
#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        T sound_speed = std::sqrt(c[i] * p[i] / rho[i]);
        c[i]          = sound_speed;
    }
}

} // namespace cooling