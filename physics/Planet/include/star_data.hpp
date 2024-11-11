//
// Created by Noah Kubli on 07.03.2024.
//

#pragma once

#include <array>
#include <iostream>

struct StarData
{
    //! @brief position of the central star
    std::array<double, 3> position{};

    //! @brief position of the central star in the last step
    std::array<double, 3> position_m1{};

    //! @brief mass of the central star
    double m{1e6};

    //! @brief inner size of the central star where particles are deleted
    double inner_size{0.};

    //! @brief Fix the position of the central star
    int fixed_star{1};

    //! @brief Constant for beta cooling in the disk
    double beta{std::numeric_limits<double>::infinity()};

    //! @brief Remove all particles with a smoothing length greater than this value
    double removal_limit_h{std::numeric_limits<double>::infinity()};

    //! @brief Don't cool any particle above this density threshold
    float cooling_rho_limit{1.683e-3};

    //! @brief Limit the timestep depending on changes in the internal energy. delta_t = K_u * u / du
    double K_u{std::numeric_limits<double>::infinity()};

    //! @brief Use a polytropic equation of state
    //! P = Kpoly * rho ^ exp_poly
    int use_polytropic_eos{1};
    //! @brief The next two parameters are used if the polytropic equation of state is activated.
    //! @brief polytropic constant
    double Kpoly{1.9998578841e-3};
    //! @brief polytropic exponent
    double exp_poly{5. / 3.};

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
        optionalIO("star::K_u", &K_u, 1);
        optionalIO("star::use_polytropic_eos", &use_polytropic_eos, 1);
        optionalIO("star::Kpoly", &Kpoly, 1);
        optionalIO("star::exp_poly", &exp_poly, 1);
    };

    //! @brief Potential from interaction between star and particles
    double potential{};

    //! @brief Values local to each rank
    //! @brief Total force acting on the star (local to rank)
    std::array<double, 3> force_local{};
    //! @brief Potential from interaction between star and particles (local to rank)
    double potential_local{};
    //! @brief Count of accreted particles (local to rank)
    size_t n_accreted_local{};
    //! @brief Count of removed particles (local to rank)
    size_t n_removed_local{};
    //! @brief Accreted mass (local to rank)
    double m_accreted_local{};
    //! @brief Removed mass (local to rank)
    double m_removed_local{};
    //! @brief Accreted momentum (local to rank)
    std::array<double, 3> p_accreted_local{};
    //! @brief Removed momentum (local to rank)
    std::array<double, 3> p_removed_local{};
    //! @brief du-timestep (local to rank)
    double t_du{};
};
