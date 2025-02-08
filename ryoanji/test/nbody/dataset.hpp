/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Generation of test input bodies
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

namespace ryoanji
{

template<class T>
static void makeCubeBodies(T* x, T* y, T* z, T* m, T* h, size_t n, double extent = 3)
{
    double ng0   = 100;
    T      hInit = std::cbrt(ng0 / n / 4.19) * extent;
    for (size_t i = 0; i < n; i++)
    {
        x[i] = drand48() * 2 * extent - extent;
        y[i] = drand48() * 2 * extent - extent;
        z[i] = drand48() * 2 * extent - extent;
        m[i] = drand48() / n;
        h[i] = hInit;
    }

    // set non-random corners
    x[0] = -extent;
    y[0] = -extent;
    z[0] = -extent;

    x[n - 1] = extent;
    y[n - 1] = extent;
    z[n - 1] = extent;
}

} // namespace ryoanji
