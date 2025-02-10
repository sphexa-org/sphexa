//
// Created by Noah Kubli on 15.03.2024.
//

#pragma once

#include <array>
#include <cstdio>

#include "accretion_impl.hpp"
#include "accretion_gpu.hpp"
#include "buffer_reduce.hpp"
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
    const auto [m_accreted, p_accreted, m_removed, p_removed] = buffered_mpi_allreduce_sum(
        star.accreted_local.mass, star.accreted_local.momentum, star.removed_local.mass, star.removed_local.momentum);

    const double m_star_new = m_accreted + star.m;

    std::array<double, 3> p_star;
    for (size_t i = 0; i < 3; i++)
    {
        p_star[i] = (star.position_m1[i] / minDt_m1) * star.m;
        p_star[i] += p_accreted[i];
        star.position_m1[i] = p_star[i] / m_star_new * minDt_m1;
    }

    star.m = m_star_new;
    if (rank == 0)
    {
        std::printf("removed mass: %g\taccreted mass: %g\tstar mass: %g\n", m_removed, m_accreted, star.m);
        std::printf("removed momentum x: %g\taccreted momentum x: %g\tstar momentum x: %g\n", p_removed[0],
                    p_accreted[0], p_star[0]);
        std::printf("removed momentum y: %g\taccreted momentum y: %g\tstar momentum y: %g\n", p_removed[1],
                    p_accreted[1], p_star[1]);
        std::printf("removed momentum z: %g\taccreted momentum z: %g\tstar momentum z: %g\n", p_removed[2],
                    p_accreted[2], p_star[2]);
        std::printf("accreted mass local: %g\n", star.accreted_local.mass);
    }
}

} // namespace disk
