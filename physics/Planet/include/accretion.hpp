//
// Created by Noah Kubli on 15.03.2024.
//

#pragma once

#include <array>
#include <cstdio>
#include <mpi.h>

#include "accretion_impl.hpp"
#include "accretion_gpu.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/primitives/accel_switch.hpp"


namespace disk
{

//! @brief Flag particles for removal. Overwrites keys for the removed particles.
template<typename Dataset, typename StarData>
void computeAccretionCondition(size_t first, size_t last, Dataset& d, StarData& star)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        computeAccretionConditionGPU(first, last, d, star);
    }
    else { computeAccretionConditionImpl(first, last, d, star); }
}

//! @brief Exchange accreted mass and momentum between ranks and add to star.
template<typename StarData>
void exchangeAndAccreteOnStar(StarData& star, double minDt_m1, int rank)
{
    double                m_accreted_global{};
    std::array<double, 3> p_accreted_global{};

    double                m_removed_global{};
    std::array<double, 3> p_removed_global{};

    MPI_Reduce(&star.accreted_local.mass, &m_accreted_global, 1, MpiType<double>{}, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(star.accreted_local.momentum.data(), p_accreted_global.data(), 3, MpiType<double>{}, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&star.removed_local.mass, &m_removed_global, 1, MpiType<double>{}, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(star.removed_local.momentum.data(), p_removed_global.data(), 3, MpiType<double>{}, MPI_SUM, 0,
               MPI_COMM_WORLD);

    if (rank == 0)
    {
        double m_star_new = m_accreted_global + star.m;

        std::array<double, 3> p_star;
        for (size_t i = 0; i < 3; i++)
        {
            p_star[i] = (star.position_m1[i] / minDt_m1) * star.m;
            p_star[i] += p_accreted_global[i];
            star.position_m1[i] = p_star[i] / m_star_new * minDt_m1;
        }

        star.m = m_star_new;

        std::printf("removed mass: %g\taccreted mass: %g\tstar mass: %g\n", m_removed_global, m_accreted_global,
                    star.m);
        std::printf("removed momentum x: %g\taccreted momentum x: %g\tstar momentum x: %g\n", p_removed_global[0],
                    p_accreted_global[0], p_star[0]);
        std::printf("removed momentum y: %g\taccreted momentum y: %g\tstar momentum y: %g\n", p_removed_global[1],
                    p_accreted_global[1], p_star[1]);
        std::printf("removed momentum z: %g\taccreted momentum z: %g\tstar momentum z: %g\n", p_removed_global[2],
                    p_accreted_global[2], p_star[2]);
    }

    if (rank == 0) { std::printf("accreted mass local: %g\n", star.accreted_local.mass); }

    MPI_Bcast(star.position_m1.data(), 3, MpiType<double>{}, 0, MPI_COMM_WORLD);
    MPI_Bcast(&star.m, 1, MpiType<double>{}, 0, MPI_COMM_WORLD);
}

} // namespace disk
