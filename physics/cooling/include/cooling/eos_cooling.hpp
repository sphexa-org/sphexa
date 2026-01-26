//
// Created by Noah Kubli on 09.07.23.
//

#pragma once

namespace cooling
{

//! @brief the maximum time-step based on local particles that Grackle can tolerate
template<class T1, class T2, typename Cooler, typename Chem>
auto coolingTimestep(size_t first, size_t last, const T1* rho, const T2* u, Cooler& cooler, Chem& chem)
{
    using CoolingFields  = typename Cooler::CoolingFields;
    const auto chemistry = cstone::getPointers(get<CoolingFields>(chem), 0);

    auto minCt = cooler.cooling_timestep(rho, u, chemistry, first, last);

    return minCt;
}

template<typename HydroData, typename ChemData, typename Cooler>
void eos_cooling(size_t startIndex, size_t endIndex, HydroData& d, ChemData& chem, Cooler& cooler)
{
    using CoolingFields  = typename Cooler::CoolingFields;
    using T              = typename HydroData::HydroType;
    const auto chemistry = cstone::getPointers(get<CoolingFields>(chem), 0);

    const auto&& rho = toHost(d.rho);
    const auto&& u   = toHost(d.u);

    std::vector<T> p(rho.size()), c(rho.size());
    cooler.computePressures(rho.data(), u.data(), chemistry, p.data(), startIndex, endIndex);

    // Write adiabatic indices into c (sound speed) first
    cooler.computeAdiabaticIndices(rho.data(), u.data(), chemistry, c.data(), startIndex, endIndex);
#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        T sound_speed = std::sqrt(c[i] * p[i] / rho[i]);
        c[i]          = sound_speed;
    }

    d.p = std::move(p);
    d.c = std::move(c);
}

} // namespace cooling