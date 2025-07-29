//
// Created by Noah Kubli on 29.11.22.
//

#pragma once

extern "C"
{
#include "grackle.h"
}

#include "cooler.hpp"

template<typename T, size_t BlockSize = 1000>
struct cooler_field_data_arr
{
    grackle_field_data data;

    std::array<int, 3>            start{0, 0, 0};
    std::array<int, 3>            end{0, 0, 0};
    std::array<int, 3>            dim{1, 0, 1};
    std::array<double, BlockSize> xv1;
    std::array<double, BlockSize> yv1;
    std::array<double, BlockSize> zv1;

    // Particle has to point to densities not fractions
    void makeGrackleFieldsFromData(T* rho, T* u, const typename cooling::Cooler<T>::GrackleFieldPtrs& particle,
                                   const size_t len)
    {
        assert(len > 0);
        assert(len <= BlockSize);
        using CoolingFields = typename cooling::Cooler<T>::CoolingFields;
        static_assert(std::is_same_v<T, gr_float>);

        std::fill(xv1.begin(), xv1.end(), 0.);
        std::fill(yv1.begin(), yv1.end(), 0.);
        std::fill(zv1.begin(), zv1.end(), 0.);

        end[1] = (int)len - 1;
        dim[1] = (int)len;

        data.grid_rank      = 2;
        data.grid_dimension = dim.data();
        data.grid_start     = start.data();
        data.grid_end       = end.data();
        data.grid_dx        = 0.0;

        data.density                  = rho;
        data.internal_energy          = u;
        data.x_velocity               = xv1.data();
        data.y_velocity               = yv1.data();
        data.z_velocity               = zv1.data();
        data.HI_density               = util::get<"HI_fraction", CoolingFields>(particle);
        data.HII_density              = util::get<"HII_fraction", CoolingFields>(particle);
        data.HM_density               = util::get<"HM_fraction", CoolingFields>(particle);
        data.HeI_density              = util::get<"HeI_fraction", CoolingFields>(particle);
        data.HeII_density             = util::get<"HeII_fraction", CoolingFields>(particle);
        data.HeIII_density            = util::get<"HeIII_fraction", CoolingFields>(particle);
        data.H2I_density              = util::get<"H2I_fraction", CoolingFields>(particle);
        data.H2II_density             = util::get<"H2II_fraction", CoolingFields>(particle);
        data.DI_density               = util::get<"DI_fraction", CoolingFields>(particle);
        data.DII_density              = util::get<"DII_fraction", CoolingFields>(particle);
        data.HDI_density              = util::get<"HDI_fraction", CoolingFields>(particle);
        data.e_density                = util::get<"e_fraction", CoolingFields>(particle);
        data.metal_density            = util::get<"metal_fraction", CoolingFields>(particle);
        data.volumetric_heating_rate  = util::get<"volumetric_heating_rate", CoolingFields>(particle);
        data.specific_heating_rate    = util::get<"specific_heating_rate", CoolingFields>(particle);
        data.RT_heating_rate          = util::get<"RT_heating_rate", CoolingFields>(particle);
        data.RT_HI_ionization_rate    = util::get<"RT_HI_ionization_rate", CoolingFields>(particle);
        data.RT_HeI_ionization_rate   = util::get<"RT_HeI_ionization_rate", CoolingFields>(particle);
        data.RT_HeII_ionization_rate  = util::get<"RT_HeII_ionization_rate", CoolingFields>(particle);
        data.RT_H2_dissociation_rate  = util::get<"RT_H2_dissociation_rate", CoolingFields>(particle);
        data.H2_self_shielding_length = util::get<"H2_self_shielding_length", CoolingFields>(particle);
    }
};
