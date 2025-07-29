//
// Created by Noah Kubli on 13.02.2024.
//

#pragma once

#include "cooler.hpp"
#include "cooler_field_data_arr.hpp"
#include "cooler_task.hpp"
#include "cooler_util.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <optional>
#include <omp.h>
#include <vector>

namespace cooling
{
template<typename T>
struct Cooler<T>::Impl
{
    friend struct Cooler<T>;

    using GrackleFieldPtrs = typename Cooler<T>::GrackleFieldPtrs;

protected:
    // Constructor made protected for Testing
    Impl()
    {
        local_initialize_chemistry_parameters(&global_values.data);
        global_values.data.grackle_data_file = &grackle_data_file_path[0];
    }

private:
    //! @brief Number of particles that are passed simultaneously to Grackle
    inline constexpr static size_t blockSize = 1000;

    //! @brief Solar mass in g
    inline constexpr static T ms_g = 1.989e33;
    //! @brief kpc in cm
    inline constexpr static T kp_cm = 3.086e21;
    //! @brief Gravitational constant in cgs units
    inline constexpr static T G_newton = 6.674e-8;
    //! @brief code unit mass in solar masses
    T m_code_in_ms = 1e16;
    //! @brief code unit length in kpc
    T l_code_in_kpc = 46400.;
    //! @brief Path to Grackle data file
    std::string grackle_data_file_path = CMAKE_SOURCE_DIR "/extern/grackle/grackle_repo/input/CloudyData_UVB=HM2012.h5";

    //! @brief Struct storing the values that need to be initialized once and passed to Grackle at each call
    struct GlobalValues
    {
        code_units             units;
        chemistry_data         data;
        chemistry_data_storage rates;
    };
    GlobalValues global_values;

    inline constexpr static std::array parameterNames{
        "m_code_in_ms", "l_code_in_kpc",
        // Grackle parameters
        "use_grackle", "with_radiative_cooling", "primordial_chemistry", "dust_chemistry", "metal_cooling",
        "UVbackground",
        //!
        //! d.char *grackle_data_file",
        "cmb_temperature_floor", "Gamma", "h2_on_dust", "use_dust_density_field", "dust_recombination_cooling",
        "photoelectric_heating", "photoelectric_heating_rate", "use_isrf_field", "interstellar_radiation_field",
        "use_volumetric_heating_rate", "use_specific_heating_rate", "three_body_rate", "cie_cooling",
        "h2_optical_depth_approximation", "ih2co", "ipiht", "HydrogenFractionByMass", "DeuteriumToHydrogenRatio",
        "SolarMetalFractionByMass", "local_dust_to_gas_ratio", "NumberOfTemperatureBins", "CaseBRecombination",
        "TemperatureStart", "TemperatureEnd", "NumberOfDustTemperatureBins", "DustTemperatureStart",
        "DustTemperatureEnd", "Compton_xray_heating", "LWbackground_sawtooth_suppression", "LWbackground_intensity",
        "UVbackground_redshift_on", "UVbackground_redshift_off", "UVbackground_redshift_fullon",
        "UVbackground_redshift_drop", "cloudy_electron_fraction_factor", "use_radiative_transfer",
        "radiative_transfer_coupled_rate_solver", "radiative_transfer_intermediate_step",
        "radiative_transfer_hydrogen_only", "self_shielding_method", "H2_self_shielding", "H2_custom_shielding",
        "h2_charge_exchange_rate", "h2_dust_rate", "h2_h_cooling_rate", "collisional_excitation_rates",
        "collisional_ionisation_rates", "recombination_cooling_rates", "bremsstrahlung_cooling_rates", "max_iterations",
        "exit_after_iterations_exceeded"};

    auto parametersTuple()
    {
        auto& d   = global_values.data;
        auto  ret = std::tie(
             m_code_in_ms, l_code_in_kpc, d.use_grackle, d.with_radiative_cooling, d.primordial_chemistry,
             d.dust_chemistry, d.metal_cooling, d.UVbackground,
             //!
             //! d.char *grackle_data_file,
             d.cmb_temperature_floor, d.Gamma, d.h2_on_dust, d.use_dust_density_field, d.dust_recombination_cooling,
             d.photoelectric_heating, d.photoelectric_heating_rate, d.use_isrf_field, d.interstellar_radiation_field,
             d.use_volumetric_heating_rate, d.use_specific_heating_rate, d.three_body_rate, d.cie_cooling,
             d.h2_optical_depth_approximation, d.ih2co, d.ipiht, d.HydrogenFractionByMass, d.DeuteriumToHydrogenRatio,
             d.SolarMetalFractionByMass, d.local_dust_to_gas_ratio, d.NumberOfTemperatureBins, d.CaseBRecombination,
             d.TemperatureStart, d.TemperatureEnd, d.NumberOfDustTemperatureBins, d.DustTemperatureStart,
             d.DustTemperatureEnd, d.Compton_xray_heating, d.LWbackground_sawtooth_suppression, d.LWbackground_intensity,
             d.UVbackground_redshift_on, d.UVbackground_redshift_off, d.UVbackground_redshift_fullon,
             d.UVbackground_redshift_drop, d.cloudy_electron_fraction_factor, d.use_radiative_transfer,
             d.radiative_transfer_coupled_rate_solver, d.radiative_transfer_intermediate_step,
             d.radiative_transfer_hydrogen_only, d.self_shielding_method, d.H2_self_shielding, d.H2_custom_shielding,
             d.h2_charge_exchange_rate, d.h2_dust_rate, d.h2_h_cooling_rate, d.collisional_excitation_rates,
             d.collisional_ionisation_rates, d.recombination_cooling_rates, d.bremsstrahlung_cooling_rates,
             d.max_iterations, d.exit_after_iterations_exceeded);

        static_assert(std::tuple_size_v<decltype(ret)> == parameterNames.size());
        return ret;
    }

    std::vector<FieldVariant> getFields()
    {
        return std::apply([](auto&... a) { return std::vector<FieldVariant>{&a...}; }, parametersTuple());
    }

    static std::vector<const char*> getParameterNames()
    {
        auto a =
            std::apply([](auto&... a) { return std::array<const char*, sizeof...(a)>{(&a[0])...}; }, parameterNames);
        return std::vector(a.begin(), a.end());
    }

    chemistry_data getDefaultChemistryData()
    {
        chemistry_data data_default;
        local_initialize_chemistry_parameters(&data_default);
        data_default.grackle_data_file = &grackle_data_file_path[0];
        return data_default;
    }

    //! @brief Initializes Grackle. Needs to be called before any calls to Grackle are made
    void init(const bool comoving_coordinates, const std::optional<T> time_unit_opt)
    {
        grackle_verbose = 1;

        // Density
        const double density_unit = m_code_in_ms * ms_g / std::pow(l_code_in_kpc * kp_cm, 3);
        // Time
        const double time_unit = time_unit_opt.value_or(std::sqrt(1. / (density_unit * G_newton)));
        // Length
        const double length_unit = l_code_in_kpc * kp_cm;
        // Velocity
        const double velocity_unit = length_unit / time_unit;

        global_values.units.density_units        = density_unit; // m_sun / (pc * pc * pc);
        global_values.units.time_units           = time_unit;    // code_time;
        global_values.units.length_units         = length_unit;  // pc;
        global_values.units.velocity_units       = velocity_unit;
        global_values.units.a_units              = 1.0;
        global_values.units.a_value              = 1.0;
        global_values.units.comoving_coordinates = comoving_coordinates ? 1 : 0;

#ifndef NDEBUG
        std::cout << "debug\n";
        std::cout << m_code_in_ms << "\t" << ms_g << "\t" << l_code_in_kpc << "\n";
        std::cout << "code units\n";
        std::cout << global_values.units.density_units << "\t" << global_values.units.time_units << "\t"
                  << global_values.units.length_units << "\n";

#endif
        global_values.data.grackle_data_file = &grackle_data_file_path[0];
        if (0 == local_initialize_chemistry_data(&global_values.data, &global_values.rates, &global_values.units))
        {
            std::cout << global_values.data.with_radiative_cooling << std::endl;
            throw std::runtime_error("Grackle: Error in _initialize_chemistry_data");
        }
    }

    void cool_particle_arr(T dt, T* rho, T* u, const GrackleFieldPtrs& particle, const size_t len)
    {
        static_assert(std::is_same_v<T, gr_float>);

        cooler_field_data_arr<T, blockSize> grackle_fields;

        grackle_fields.makeGrackleFieldsFromData(rho, u, particle, len);

        auto ret_value = local_solve_chemistry(&global_values.data, &global_values.rates, &global_values.units,
                                               &grackle_fields.data, dt);
        if (ret_value == 0) { throw std::runtime_error("Grackle: Error in local_solve_chemistry"); }
    }

    void compute_temperature_arr(T* rho, T* u, const GrackleFieldPtrs& particle, T* temp, const size_t len)
    {
        static_assert(std::is_same_v<T, gr_float>);

        cooler_field_data_arr<T, blockSize> grackle_fields;

        grackle_fields.makeGrackleFieldsFromData(rho, u, particle, len);

        if (0 == local_calculate_temperature(&global_values.data, &global_values.rates, &global_values.units,
                                             &grackle_fields.data, temp))
        {
            throw std::runtime_error("Grackle: Error in local_calculate_temperature");
        }
    }

    void pressure_arr(T* rho, T* u, const GrackleFieldPtrs& particle, T* p, const size_t len)
    {
        static_assert(std::is_same_v<T, gr_float>);

        cooler_field_data_arr<T, blockSize> grackle_fields;

        grackle_fields.makeGrackleFieldsFromData(rho, u, particle, len);
        auto ret_value = local_calculate_pressure(&global_values.data, &global_values.rates, &global_values.units,
                                                  &grackle_fields.data, p);
        if (ret_value == 0) { throw std::runtime_error("Grackle: local_calculate_pressure"); }
    }

    void cooling_time_arr(T* rho, T* u, const GrackleFieldPtrs& particle, T* ct, const size_t len)
    {
        static_assert(std::is_same_v<T, gr_float>);

        cooler_field_data_arr<T, blockSize> grackle_fields;

        grackle_fields.makeGrackleFieldsFromData(rho, u, particle, len);

        auto ret_value = local_calculate_cooling_time(&global_values.data, &global_values.rates, &global_values.units,
                                                      &grackle_fields.data, ct);
        if (ret_value == 0) { throw std::runtime_error("Grackle: local_calculate_cooling_time"); }
    }

    void adiabatic_index_arr(T* rho, T* u, const GrackleFieldPtrs& particle, T* gamma, const size_t len)
    {
        static_assert(std::is_same_v<T, gr_float>);

        cooler_field_data_arr<T, blockSize> grackle_fields;

        grackle_fields.makeGrackleFieldsFromData(rho, u, particle, len);

        auto ret_value = local_calculate_gamma(&global_values.data, &global_values.rates, &global_values.units,
                                               &grackle_fields.data, gamma);
        if (ret_value == 0) { throw std::runtime_error("Grackle: local_calculate_gamma"); }
    }

    template<typename Trho, typename Tu>
    void cool_particles(const T dt, const Trho* rho, const Tu* u, const GrackleFieldPtrs& particle, Tu* du,
                        const size_t first, const size_t last)
    {
        const Partition<blockSize> partition(first, last);

        const auto compute_du = [&](const auto& u_block, const Task& b)
        {
            for (size_t i = 0; i < b.len; i++)
            {
                du[i + b.first] += (u_block[i] - u[i + b.first]) / dt;
            }
        };

#pragma omp parallel for
        for (size_t i = 0; i < partition.n_bins; i++)
        {
            Task task(i, partition);
            auto gblock = extractBlock(rho, u, particle, task.first, task.last);

            cool_particle_arr(dt, gblock.rho.data(), gblock.u.data(), cstone::getPointers(gblock.grackleFields, 0),
                              task.len);

            storeBlock(gblock, particle, task.first, task.last);
            compute_du(gblock.u, task);
        }
    }

    template<typename Trho, typename Tu, typename Tp>
    void computeTemperature(const Trho* rho, const Tu* u, const GrackleFieldPtrs& particle, Tp* temp, size_t first,
                            size_t last)
    {
        const Partition<blockSize> partition(first, last);
#pragma omp parallel for
        for (size_t i = 0; i < partition.n_bins; i++)
        {
            Task task(i, partition);
            auto gblock = extractBlock(rho, u, particle, task.first, task.last);

            std::array<T, blockSize> temp_ret;
            compute_temperature_arr(gblock.rho.data(), gblock.u.data(), cstone::getPointers(gblock.grackleFields, 0),
                                    temp_ret.data(), task.len);
            std::copy_n(temp_ret.data(), task.len, temp + task.first);
        }
    }

    template<typename Trho, typename Tu, typename Tp>
    void computePressures(const Trho* rho, const Tu* u, const GrackleFieldPtrs& particle, Tp* p, size_t first,
                          size_t last)
    {
        const Partition<blockSize> partition(first, last);

#pragma omp parallel for
        for (size_t i = 0; i < partition.n_bins; i++)
        {
            Task task(i, partition);
            auto gblock = extractBlock(rho, u, particle, task.first, task.last);

            std::array<T, blockSize> p_ret;
            pressure_arr(gblock.rho.data(), gblock.u.data(), cstone::getPointers(gblock.grackleFields, 0), p_ret.data(),
                         task.len);
            std::copy_n(p_ret.data(), task.len, p + task.first);
        }
    }

    template<typename Trho, typename Tu, typename Tgamma>
    void computeAdiabaticIndices(const Trho* rho, const Tu* u, const GrackleFieldPtrs& particle, Tgamma* gamma,
                                 size_t first, size_t last)
    {
        const Partition<blockSize> partition(first, last);

#pragma omp parallel for
        for (size_t i = 0; i < partition.n_bins; i++)
        {
            Task task(i, partition);
            auto gblock = extractBlock(rho, u, particle, task.first, task.last);

            std::array<T, blockSize> gamma_ret;
            adiabatic_index_arr(gblock.rho.data(), gblock.u.data(), cstone::getPointers(gblock.grackleFields, 0),
                                gamma_ret.data(), task.len);
            std::copy_n(gamma_ret.data(), task.len, gamma + task.first);
        }
    }

    template<typename Trho, typename Tu>
    double min_cooling_time(const Trho* rho, const Tu* u, const GrackleFieldPtrs& particle, size_t first, size_t last)
    {
        const Partition<blockSize> partition(first, last);

        double ct_min = std::numeric_limits<double>::infinity();

#pragma omp parallel for reduction(min : ct_min)
        for (size_t i = 0; i < partition.n_bins; i++)
        {
            Task task(i, partition);
            auto gblock = extractBlock(rho, u, particle, task.first, task.last);

            std::array<T, blockSize> ct_ret;
            cooling_time_arr(gblock.rho.data(), gblock.u.data(), cstone::getPointers(gblock.grackleFields, 0),
                             ct_ret.data(), task.len);

            std::transform(ct_ret.begin(), ct_ret.end(), ct_ret.begin(), [](double a) { return std::abs(a); });

            const double ct = *std::min_element(ct_ret.begin(), ct_ret.begin() + task.len);
            ct_min          = std::min(ct_min, ct);
        }
        return ct_min;
    }

protected:
    using GrackleFieldsArray =
        util::Reduce<std::tuple, util::Repeat<util::TypeList<std::array<double, blockSize>>, numFields>>;

    struct GrackleBlock
    {
        std::array<double, blockSize> rho, u;
        GrackleFieldsArray            grackleFields;
    };

    //! @brief Extract range [first:last] of arguments into [0:last-first] of the output, convert fractions to densities
    template<typename Trho, typename Tu>
    GrackleBlock extractBlock(const Trho* rho, const Tu* u, const GrackleFieldPtrs& grFields, size_t first, size_t last)
    {
        GrackleBlock block;
        std::copy(rho + first, rho + last, block.rho.begin());
        std::copy(u + first, u + last, block.u.begin());

        auto fractionsSrc   = util::get<Fractions, CoolingFields>(grFields);
        auto fractionsBlock = util::get<Fractions, CoolingFields>(cstone::getPointers(block.grackleFields, 0));
        for_each_tuples([first, last, rho](auto* src, auto* dest)
                        { std::transform(src + first, src + last, rho + first, dest, std::multiplies<>{}); },
                        fractionsSrc, fractionsBlock);

        auto ratesSrc   = util::get<Rates, CoolingFields>(grFields);
        auto ratesBlock = util::get<Rates, CoolingFields>(cstone::getPointers(block.grackleFields, 0));
        for_each_tuples([first, last](auto* src, auto* dest) { std::copy(src + first, src + last, dest); }, ratesSrc,
                        ratesBlock);

        return block;
    }

    //! @brief Store @p Task data from [0:last-first] into grFields [first:last], converting densities to fractions
    void storeBlock(const GrackleBlock& block, const GrackleFieldPtrs& grFields, size_t first, size_t last)
    {
        auto densitiesBlock = util::get<Fractions, CoolingFields>(cstone::getPointers(block.grackleFields, 0));
        auto fractionsDest  = util::get<Fractions, CoolingFields>(grFields);
        for_each_tuples([first, last, rho = block.rho.data()](const auto* src, auto* dest)
                        { std::transform(src, src + last - first, rho, dest + first, std::divides<>{}); },
                        densitiesBlock, fractionsDest);

        auto ratesBlock = util::get<Rates, CoolingFields>(cstone::getPointers(block.grackleFields, 0));
        auto ratesDest  = util::get<Rates, CoolingFields>(grFields);
        for_each_tuples([first, last](auto* src, auto* dest) { std::copy(src, src + last - first, dest + first); },
                        ratesBlock, ratesDest);
    }
};
} // namespace cooling
