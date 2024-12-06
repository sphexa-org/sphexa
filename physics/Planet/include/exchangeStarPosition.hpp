//
// Created by Noah Kubli on 04.03.2024.
//

#pragma once

#include <mpi.h>
#include "cstone/primitives/mpi_wrappers.hpp"

namespace disk
{

//! @brief Compute the new star position by exchanging the force between the nodes and integrating the acceleration
template<typename StarData>
void computeAndExchangeStarPosition(StarData& star, double dt, double dt_m1, int rank)
{
    if (star.fixed_star == 1) { return; }
    std::array<double, 4> global_force{};

    MPI_Reduce(star.force_local.data(), global_force.data(), 4, MpiType<double>{}, MPI_SUM, 0, MPI_COMM_WORLD);
//    MPI_Reduce(&star.potential_local, &star.potential, 1, MpiType<double>{}, MPI_SUM, 0, MPI_COMM_WORLD);
    star.potential = global_force[0];

    if (rank == 0)
    {
        double a_starx = global_force[1] / star.m;
        double a_stary = global_force[2] / star.m;
        double a_starz = global_force[3] / star.m;

        auto integrate = [dt, dt_m1](double a, double x_m1)
        {
            double deltaB = 0.5 * (dt + dt_m1);
            auto   Val    = x_m1 * (1. / dt_m1);
            auto   dx     = dt * Val + a * deltaB * dt;
            return dx;
        };

        double dx = integrate(a_starx, star.position_m1[0]);
        double dy = integrate(a_stary, star.position_m1[1]);
        double dz = integrate(a_starz, star.position_m1[2]);
        star.position[0] += dx;
        star.position[1] += dy;
        star.position[2] += dz;
        star.position_m1[0] = dx;
        star.position_m1[1] = dy;
        star.position_m1[2] = dz;
    }

    MPI_Bcast(star.position.data(), 3, MpiType<double>{}, 0, MPI_COMM_WORLD);
    MPI_Bcast(star.position_m1.data(), 3, MpiType<double>{}, 0, MPI_COMM_WORLD);
}
} // namespace disk
