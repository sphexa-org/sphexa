
#include <iostream>
#include <vector>
#include <cmath>
#include "gtest/gtest.h"

#include "cooling/cooler.hpp"
#include "cooling/chemistry_data.hpp"
#include "cooling/cooler_impl.hpp"
#include "cooling/cooler_task.hpp"

#include "cstone/fields/field_get.hpp"
#include "io/ifile_io.hpp"

#include "init/settings.hpp"

TEST(cooling_grackle, testCoolParticles)
{
    using Real = double;

    constexpr Real density_units = 1.67e-24;
    constexpr Real time_units    = 1.0e12;
    constexpr Real length_units  = 1.0;

    // constexpr Real GCGS = 6.674e-8;

    constexpr Real density_units_c = 1. / (time_units * time_units);
    // EXPECT_NEAR(density_units, density_units_c, 1e-26);
    printf("density units: %g\t%g\n", density_units, density_units_c);
    constexpr double MSOLG = 1.989e33;
    const double     KPCCM = 3.086e21;

    const Real mass_unit = std::pow(length_units, 3.0) * density_units / MSOLG;

    cooling::Cooler<Real> cd;

    std::map<std::string, double> grackleOptions;
    grackleOptions["cooling::m_code_in_ms"]           = mass_unit;
    grackleOptions["cooling::l_code_in_kpc"]          = 1. / KPCCM;
    grackleOptions["cooling::use_grackle"]            = 1;
    grackleOptions["cooling::with_radiative_cooling"] = 1;
    grackleOptions["cooling::primordial_chemistry"]   = 3;
    grackleOptions["cooling::dust_chemistry"]         = 1;
    grackleOptions["cooling::UVbackground"]           = 1;
    grackleOptions["cooling::metal_cooling"]          = 1;

    sphexa::BuiltinWriter extractor(grackleOptions);
    cd.loadOrStoreAttributes(&extractor);

    cd.init(0, time_units);

    constexpr Real tiny_number = 1.e-20;
    constexpr Real dt          = 3.15e7 * 1e6; // grackle_units.time_units;
    constexpr Real mh          = 1.67262171e-24;
    constexpr Real kboltz      = 1.3806504e-16;

    auto rho = std::vector<float>{1.1, 1.0};
    /*Real temperature_units =
            mh *
            pow(cd.get_global_values().units.a_units * cd.get_global_values().units.length_units /
       cd.get_global_values().units.time_units, 2.) / kboltz;*/

    Real temperature_units = mh * std::pow(length_units / time_units, 2.) / kboltz;

    auto u                        = std::vector<Real>{1000. / temperature_units, 1000. / temperature_units};
    auto HI_fraction              = std::vector<Real>{0.76, 0.76};
    auto HII_fraction             = std::vector<Real>{tiny_number, tiny_number};
    auto HM_fraction              = std::vector<Real>{tiny_number, tiny_number};
    auto HeI_fraction             = std::vector<Real>{0.24, 0.24};
    auto HeII_fraction            = std::vector<Real>{tiny_number, tiny_number};
    auto HeIII_fraction           = std::vector<Real>{tiny_number, tiny_number};
    auto H2I_fraction             = std::vector<Real>{tiny_number, tiny_number};
    auto H2II_fraction            = std::vector<Real>{tiny_number, tiny_number};
    auto DI_fraction              = std::vector<Real>{2.0 * 3.4e-5, 2.0 * 3.4e-5};
    auto DII_fraction             = std::vector<Real>{tiny_number, tiny_number};
    auto HDI_fraction             = std::vector<Real>{tiny_number, tiny_number};
    auto e_fraction               = std::vector<Real>{tiny_number, tiny_number};
    auto metal_fraction           = std::vector<Real>{0.01295, 0.01295};
    auto volumetric_heating_rate  = std::vector<Real>{0., 0.};
    auto specific_heating_rate    = std::vector<Real>{0., 0.};
    auto RT_heating_rate          = std::vector<Real>{0., 0.};
    auto RT_HI_ionization_rate    = std::vector<Real>{0., 0.};
    auto RT_HeI_ionization_rate   = std::vector<Real>{0., 0.};
    auto RT_HeII_ionization_rate  = std::vector<Real>{0., 0.};
    auto RT_H2_dissociation_rate  = std::vector<Real>{0., 0.};
    auto H2_self_shielding_length = std::vector<Real>{0., 0.};

    std::cout << HI_fraction[0] << std::endl;
    std::cout << HeI_fraction[0] << std::endl;
    std::cout << metal_fraction[0] << std::endl;

    auto grData =
        std::tie(HI_fraction, HII_fraction, HM_fraction, HeI_fraction, HeII_fraction, HeIII_fraction, H2I_fraction,
                 H2II_fraction, DI_fraction, DII_fraction, HDI_fraction, e_fraction, metal_fraction,
                 volumetric_heating_rate, specific_heating_rate, RT_heating_rate, RT_HI_ionization_rate,
                 RT_HeI_ionization_rate, RT_HeII_ionization_rate, RT_H2_dissociation_rate, H2_self_shielding_length);

    std::vector<Real> du(2);
    cd.cool_particles(dt / time_units, rho.data(), u.data(), cstone::getPointers(grData, 0), du.data(), 0, 2);

    std::cout << HI_fraction[0] << std::endl;
    std::cout << HI_fraction[1] << std::endl;

    EXPECT_NEAR(HI_fraction[0], 0.640295, 1e-6);
    EXPECT_NEAR(HI_fraction[1], 0.630705, 1e-6);
    u[1] = u[1] + (dt / time_units) * du[1];
    EXPECT_NEAR(u[1], 2.95159e+35, 1e30);
}

// This test just produces a table of cooling values for different choices of rho and u
TEST(cooling_grackle2, test2)
{
    // Path where to write the table
    const std::string writePath{"sphexa_cooling_test.txt"};

    using Real = double;
    cooling::Cooler<Real>         cd;
    std::map<std::string, double> grackleOptions;
    grackleOptions["cooling::m_code_in_ms"]           = 1e16;
    grackleOptions["cooling::l_code_in_kpc"]          = 46400;
    grackleOptions["cooling::use_grackle"]            = 1;
    grackleOptions["cooling::with_radiative_cooling"] = 1;
    grackleOptions["cooling::primordial_chemistry"]   = 1;
    grackleOptions["cooling::dust_chemistry"]         = 0;
    grackleOptions["cooling::UVbackground"]           = 0;
    grackleOptions["cooling::metal_cooling"]          = 0;

    sphexa::BuiltinWriter attributeSetter(grackleOptions);
    cd.loadOrStoreAttributes(&attributeSetter);
    cd.init(0);

    constexpr Real tiny_number = 1.e-20;
    constexpr Real dt          = 0.01; // grackle_units.time_units;

    size_t n_rho       = 100;
    size_t n_u         = 100;
    Real   rho_min_log = -2;
    Real   rho_max_log = 3;
    Real   u_min_log   = -3;
    Real   u_max_log   = 1.5;

    std::vector<Real> rho_vec(n_rho);
    std::vector<Real> u_vec(n_u);
    for (size_t i = 0; i < n_rho; i++)
    {
        Real val   = (rho_max_log - rho_min_log) / n_rho * i + rho_min_log;
        rho_vec[i] = std::pow(10., val);
    }
    for (size_t i = 0; i < n_u; i++)
    {
        Real val = (u_max_log - u_min_log) / n_u * i + u_min_log;
        u_vec[i] = std::pow(10., val);
    }

    auto cool_test_data = [&dt, &cd](Real rho_in, Real u_in)
    {
        auto rho                      = std::vector<float>{float(rho_in)};
        auto u                        = std::vector<Real>{u_in};
        auto HI_fraction              = std::vector<Real>{0.76};
        auto HII_fraction             = std::vector<Real>{tiny_number};
        auto HM_fraction              = std::vector<Real>{tiny_number};
        auto HeI_fraction             = std::vector<Real>{0.24};
        auto HeII_fraction            = std::vector<Real>{tiny_number};
        auto HeIII_fraction           = std::vector<Real>{tiny_number};
        auto H2I_fraction             = std::vector<Real>{tiny_number};
        auto H2II_fraction            = std::vector<Real>{tiny_number};
        auto DI_fraction              = std::vector<Real>{2.0 * 3.4e-5};
        auto DII_fraction             = std::vector<Real>{tiny_number};
        auto HDI_fraction             = std::vector<Real>{tiny_number};
        auto e_fraction               = std::vector<Real>{tiny_number};
        auto metal_fraction           = std::vector<Real>{tiny_number};
        auto volumetric_heating_rate  = std::vector<Real>{0.};
        auto specific_heating_rate    = std::vector<Real>{0.};
        auto RT_heating_rate          = std::vector<Real>{0.};
        auto RT_HI_ionization_rate    = std::vector<Real>{0.};
        auto RT_HeI_ionization_rate   = std::vector<Real>{0.};
        auto RT_HeII_ionization_rate  = std::vector<Real>{0.};
        auto RT_H2_dissociation_rate  = std::vector<Real>{0.};
        auto H2_self_shielding_length = std::vector<Real>{0.};

        auto grData = std::tie(HI_fraction, HII_fraction, HM_fraction, HeI_fraction, HeII_fraction, HeIII_fraction,
                               H2I_fraction, H2II_fraction, DI_fraction, DII_fraction, HDI_fraction, e_fraction,
                               metal_fraction, volumetric_heating_rate, specific_heating_rate, RT_heating_rate,
                               RT_HI_ionization_rate, RT_HeI_ionization_rate, RT_HeII_ionization_rate,
                               RT_H2_dissociation_rate, H2_self_shielding_length);

        std::vector<Real> du(1);
        cd.cool_particles(dt, &rho[0], &u[0], cstone::getPointers(grData, 0), &du[0], 0, 1);

        return u[0] + dt * du[0];
    };
    std::vector<Real> results(n_rho * n_u);
    std::FILE*        file = std::fopen(writePath.c_str(), "w");
    if (!file) throw std::runtime_error("File could not be opened");
    for (size_t i = 0; i < n_rho; i++)
    {
        for (size_t k = 0; k < n_u; k++)
        {
            // size_t it = k + i * n_u;
            Real u_cooled = cool_test_data(rho_vec[i], u_vec[k]);
            std::fprintf(file, "%g %g %g\n", rho_vec[i], u_vec[k], u_cooled);
        }
    }
    std::fclose(file);
}

static bool checkVectorDifferences(const auto& vec1, const auto& vec2, const double error = 1e-6)
{
    size_t s = vec1.size();
    for (size_t i = 0; i < s; i++)
    {
        if (std::abs(vec1[i] - vec2[i]) > error) { return false; }
    }
    return true;
}

struct CoolerDataTest : public ::testing::Test, cooling::Cooler<double>::Impl
{
    using CoolingFields = cooling::Cooler<double>::CoolingFields;

    cooling::ChemistryData<double> chemistry;
    cooling::ChemistryData<double> chemistry_copy;
    std::vector<double>            rho, u;

    inline static constexpr size_t       block_size = 100;
    inline static constexpr size_t       data_size  = 1031;
    const cooling::Partition<block_size> partition;

    CoolerDataTest()
        : partition(0, data_size)
    {
    }

    //! @brief Fill some values in chemistry and rho and copy chemistry into chemistry_copy
    void SetUp() override
    {
        std::apply([this](auto... f) { chemistry.setConserved(f.value...); },
                   make_tuple(cooling::Cooler<double>::CoolingFields{}));

        chemistry.resize(data_size);
        rho.resize(data_size);
        u.resize(data_size);

        auto fill_ascending = [](auto& vec, double start, double end)
        {
            for (size_t i = 0; i < vec.size(); i++)
            {
                vec[i] = start + i * (end - start) / vec.size();
            }
        };

        fill_ascending(get<"HI_fraction">(chemistry), 1e-3, 0.2);
        fill_ascending(get<"HDI_fraction">(chemistry), 0.1, 1.);
        fill_ascending(get<"RT_heating_rate">(chemistry), 1e-3, 1.);
        fill_ascending(get<"specific_heating_rate">(chemistry), 0.8, 1.);
        fill_ascending(rho, 1., 150.);
        fill_ascending(u, 150., 300.);
        chemistry_copy = chemistry;
    }
};

//! @brief Call convertToDens on Chemistry_copy, see if rates are untouched
TEST_F(CoolerDataTest, extractBlock)
{
    auto taskVecEqualConvert = [](const auto& task_vec, const auto& vec, const auto& rho, const cooling::Task task)
    {
        std::vector<double> species_density(task.len);
        std::transform(vec.begin() + task.first, vec.begin() + task.last, rho.begin() + task.first,
                       species_density.begin(), std::multiplies<>{});

        return std::equal(task_vec.begin(), task_vec.begin() + task.len, species_density.begin());
    };

    for (size_t i = 0; i < partition.n_bins; i++)
    {
        cooling::Task task(i, partition);
        GrackleBlock  gblock = extractBlock(
             rho.data(), u.data(), cstone::getPointers(get<CoolingFields>(chemistry_copy), 0), task.first, task.last);

        EXPECT_TRUE(std::equal(gblock.rho.begin(), gblock.rho.begin() + task.len, rho.begin() + task.first));
        EXPECT_TRUE(std::equal(gblock.u.begin(), gblock.u.begin() + task.len, u.begin() + task.first));

        auto rt_rate = util::get<"RT_heating_rate", CoolingFields>(gblock.grackleFields);
        EXPECT_TRUE(std::equal(rt_rate.begin(), rt_rate.begin() + task.len,
                               get<"RT_heating_rate">(chemistry).begin() + task.first));

        auto sh_rate = util::get<"specific_heating_rate", CoolingFields>(gblock.grackleFields);
        EXPECT_TRUE(std::equal(sh_rate.begin(), sh_rate.begin() + task.len,
                               get<"specific_heating_rate">(chemistry).begin() + task.first));

        EXPECT_TRUE(taskVecEqualConvert(util::get<"HI_fraction", CoolingFields>(gblock.grackleFields),
                                        get<"HI_fraction">(chemistry), rho, task));
        EXPECT_TRUE(taskVecEqualConvert(util::get<"HDI_fraction", CoolingFields>(gblock.grackleFields),
                                        get<"HDI_fraction">(chemistry), rho, task));
    }
}

//! @brief Call convertToDens on Chemistry_copy, then convert back and see if fractions are the same
TEST_F(CoolerDataTest, storeBlock)
{
    for (size_t i = 0; i < partition.n_bins; i++)
    {
        cooling::Task b(i, partition);
        GrackleBlock  gblock;

        std::copy_n(rho.data() + b.first, b.len, gblock.rho.data());

        std::copy_n(get<"RT_heating_rate">(chemistry).data() + b.first, b.len,
                    util::get<"RT_heating_rate", CoolingFields>(gblock.grackleFields).data());
        std::copy_n(get<"specific_heating_rate">(chemistry).data() + b.first, b.len,
                    util::get<"specific_heating_rate", CoolingFields>(gblock.grackleFields).data());

        auto* hI_frac = util::get<"HI_fraction", CoolingFields>(gblock.grackleFields).data();
        std::copy_n(get<"HI_fraction">(chemistry).data() + b.first, b.len, hI_frac);
        std::transform(hI_frac, hI_frac + b.len, rho.data() + b.first, hI_frac, std::multiplies<>{});

        auto* hDI_frac = util::get<"HDI_fraction", CoolingFields>(gblock.grackleFields).data();
        std::copy_n(get<"HDI_fraction">(chemistry).data() + b.first, b.len, hDI_frac);
        std::transform(hDI_frac, hDI_frac + b.len, rho.data() + b.first, hDI_frac, std::multiplies<>{});

        storeBlock(gblock, cstone::getPointers(get<CoolingFields>(chemistry_copy), 0), b.first, b.last);
    }
    EXPECT_TRUE(
        checkVectorDifferences(get<"RT_heating_rate">(chemistry), get<"RT_heating_rate">(chemistry_copy), 1e-9));
    EXPECT_TRUE(checkVectorDifferences(get<"specific_heating_rate">(chemistry),
                                       get<"specific_heating_rate">(chemistry_copy), 1e-9));
    EXPECT_TRUE(checkVectorDifferences(get<"HI_fraction">(chemistry), get<"HI_fraction">(chemistry_copy), 1e-9));
    EXPECT_TRUE(checkVectorDifferences(get<"HDI_fraction">(chemistry), get<"HDI_fraction">(chemistry_copy), 1e-9));
}
