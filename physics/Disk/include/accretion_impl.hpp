//
// Created by Noah Kubli on 14.03.2024.
//

#pragma once

#include "cstone/tree/definitions.h"
#include "removalStatistics.hpp"
#include "star_data.hpp"

namespace disk
{

template<typename Dataset>
void computeAccretionConditionImpl(size_t first, size_t last, Dataset& d, StarData& star)
{
    const double star_size2 = star.inner_size * star.inner_size;

    RemovalStatistics accreted_local{}, removed_local{};

    auto markForRemovalAndAdd = [&d](RemovalStatistics& statistics, size_t i)
    {
        d.keys[i]  = cstone::removeKey<typename Dataset::KeyType>::value;
        statistics = statistics + RemovalStatistics{d.m[i], {d.m[i] * d.vx[i], d.m[i] * d.vy[i], d.m[i] * d.vz[i]}, 1};
    };

#pragma omp declare reduction(add_statistics:RemovalStatistics : omp_out = omp_out + omp_in) initializer(omp_priv = {})

#pragma omp parallel for reduction(add_statistics : accreted_local, removed_local)
    for (size_t i = first; i < last; i++)
    {
        const double dx    = d.x[i] - star.position[0];
        const double dy    = d.y[i] - star.position[1];
        const double dz    = d.z[i] - star.position[2];
        const double dist2 = dx * dx + dy * dy + dz * dz;

        if (dist2 < star_size2) { markForRemovalAndAdd(accreted_local, i); }
        else if (d.h[i] > star.removal_limit_h) { markForRemovalAndAdd(removed_local, i); }
    }

    star.accreted_local = accreted_local;
    star.removed_local  = removed_local;
}

} // namespace disk
