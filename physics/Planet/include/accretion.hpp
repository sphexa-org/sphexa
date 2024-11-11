//
// Created by Noah Kubli on 15.03.2024.
//

#pragma once

#include <cassert>
#include <mpi.h>
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/fields/field_get.hpp"
#include "cstone/tree/accel_switch.hpp"
#include "cstone/cuda/cuda_stubs.h"
#include "cstone/util/type_list.hpp"

#include "sph/particles_data.hpp"

#include "accretion_impl.hpp"
#include "accretion_gpu.hpp"

namespace planet
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

    MPI_Reduce(&star.m_accreted_local, &m_accreted_global, 1, MpiType<double>{}, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(star.p_accreted_local.data(), p_accreted_global.data(), 3, MpiType<double>{}, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&star.m_removed_local, &m_removed_global, 1, MpiType<double>{}, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(star.p_removed_local.data(), p_removed_global.data(), 3, MpiType<double>{}, MPI_SUM, 0, MPI_COMM_WORLD);

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

        printf("removed mass: %g\taccreted mass: %g\tstar mass: %g\n", m_removed_global, m_accreted_global, star.m);
        printf("removed momentum x: %g\taccreted momentum x: %g\tstar momentum x: %g\n", p_removed_global[0],
               p_accreted_global[0], p_star[0]);
        printf("removed momentum y: %g\taccreted momentum y: %g\tstar momentum y: %g\n", p_removed_global[1],
               p_accreted_global[1], p_star[1]);
        printf("removed momentum z: %g\taccreted momentum z: %g\tstar momentum z: %g\n", p_removed_global[2],
               p_accreted_global[2], p_star[2]);
    }

    if (rank == 0) { printf("accreted mass local: %g\n", star.m_accreted_local); }

    MPI_Bcast(star.position_m1.data(), 3, MpiType<double>{}, 0, MPI_COMM_WORLD);
    MPI_Bcast(&star.m, 1, MpiType<double>{}, 0, MPI_COMM_WORLD);
}

} // namespace planet
