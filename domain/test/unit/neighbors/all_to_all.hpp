/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief All-to-all neighbor search for use in tests as reference
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>

#include "cstone/findneighbors.hpp"

namespace cstone
{

//! @brief simple N^2 all-to-all neighbor search
template<class T>
static void all2allNeighbors(const T* x,
                             const T* y,
                             const T* z,
                             const T* h,
                             LocalIndex n,
                             LocalIndex* neighbors,
                             unsigned* neighborsCount,
                             unsigned ngmax,
                             const Box<T>& box)
{
#pragma omp parallel for
    for (LocalIndex i = 0; i < n; ++i)
    {
        T radius = 2 * h[i];
        T r2     = radius * radius;

        T xi = x[i], yi = y[i], zi = z[i];

        unsigned ngcount = 0;
        for (LocalIndex j = 0; j < n; ++j)
        {
            if (j == i) { continue; }
            if (distanceSq<true>(xi, yi, zi, x[j], y[j], z[j], box) < r2)
            {
                if (ngcount < ngmax) { neighbors[i * ngmax + ngcount] = j; }
                ngcount++;
            }
        }
        neighborsCount[i] = ngcount;
    }
}

static void sortNeighbors(LocalIndex* neighbors, unsigned* neighborsCount, LocalIndex n, unsigned ngmax)
{
    for (LocalIndex i = 0; i < n; ++i)
    {
        std::sort(neighbors + i * ngmax, neighbors + i * ngmax + neighborsCount[i]);
    }
}

} // namespace cstone
