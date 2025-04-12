//
// Created by Noah Kubli on 07.03.2024.
//

#pragma once

#include <array>
#include <limits>
#include <iostream>

#include "cstone/tree/definitions.h"
#include "removal_statistics.hpp"

namespace disk
{

struct StarData
{
    //! @brief position of the central star
    cstone::Vec3<double> position{};

    //! @brief position of the central star in the last step
    cstone::Vec3<double> position_m1{};

    //! @brief mass of the central star
    double m{1e6};

    //! @brief inner size of the central star where particles are accreted
    double inner_size{0.};

    //! @brief Fix the position of the central star instead of integrating the position
    int fixed_star{1};

    //! @brief Constant for beta cooling in the disk
    double beta{std::numeric_limits<double>::infinity()};

    //! @brief Remove all particles with a smoothing length greater than this value
    double removal_limit_h{std::numeric_limits<double>::infinity()};

    //! @brief Don't cool any particle above this density threshold
    double cooling_rho_limit{std::numeric_limits<float>::infinity()};

    //! @brief Don't cool any particle whose internal energy is below
    double u_floor{0.};

    //! @brief Limit the timestep depending on changes in the internal energy. delta_t = K_u * u / du
    double K_u{std::numeric_limits<double>::infinity()};

    template<typename Archive>
    void loadOrStoreAttributes(Archive* ar)
    {
        //! @brief load or store an attribute, skips non-existing attributes on load.
        auto optionalIO = [ar](const std::string& attribute, auto* location, size_t attrSize)
        {
            try
            {
                ar->stepAttribute(attribute, location, attrSize);
            }
            catch (std::out_of_range&)
            {
                if (ar->rank() == 0)
                {
                    std::cout << "Attribute " << attribute
                              << " not set in file or initializer, setting to default value " << *location << std::endl;
                }
            }
        };

        optionalIO("star::x", &position[0], 1);
        optionalIO("star::y", &position[1], 1);
        optionalIO("star::z", &position[2], 1);
        optionalIO("star::x_m1", &position_m1[0], 1);
        optionalIO("star::y_m1", &position_m1[1], 1);
        optionalIO("star::z_m1", &position_m1[2], 1);
        optionalIO("star::m", &m, 1);
        optionalIO("star::inner_size", &inner_size, 1);
        optionalIO("star::fixed_star", &fixed_star, 1);
        optionalIO("star::beta", &beta, 1);
        optionalIO("star::removal_limit_h", &removal_limit_h, 1);
        optionalIO("star::cooling_rho_limit", &cooling_rho_limit, 1);
        optionalIO("star::u_floor", &u_floor, 1);
        optionalIO("star::K_u", &K_u, 1);
    };

    //! @brief Potential from interaction between star and particles
    double potential{};

    //! @brief Values local to each rank
    //! @brief Total potential [0] and force [1..3] acting on the star (local to rank)
    cstone::Vec4<double> force_local{};

    //! @brief Statistics of accreted particles
    RemovalStatistics accreted_local;

    //! @brief Statistics of removed particles
    RemovalStatistics removed_local;

    //! @brief du-timestep (local to rank)
    double t_du{};
};
} // namespace disk
