/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Direct kernel comparison against the CPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/thrust_util.cuh"

#include "dataset.hpp"
#include "ryoanji/nbody/direct.cuh"
#include "ryoanji/nbody/traversal_cpu.hpp"

using namespace ryoanji;

TEST(DirectSum, MatchCpu)
{
    using T          = double;
    size_t numBodies = 1000;
    T      boxLength = 3;
    int    numShells = 1;

    std::vector<T> x(numBodies), y(numBodies), z(numBodies), m(numBodies), h(numBodies);
    ryoanji::makeCubeBodies(x.data(), y.data(), z.data(), m.data(), h.data(), numBodies, boxLength);

    // upload to device
    thrust::device_vector<T> d_x = x, d_y = y, d_z = z, d_m = m, d_h = h;
    thrust::device_vector<T> p(numBodies), ax(numBodies), ay(numBodies), az(numBodies);

    Vec3<T> box{boxLength, boxLength, boxLength};
    directSum(0, numBodies, numBodies, box, numShells, rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_m), rawPtr(d_h),
              rawPtr(p), rawPtr(ax), rawPtr(ay), rawPtr(az));

    // download body accelerations
    thrust::host_vector<T> h_p = p, h_ax = ax, h_ay = ay, h_az = az;

    T G = 1.0;

    std::vector<T> refP(numBodies), refAx(numBodies), refAy(numBodies), refAz(numBodies);
    directSum(x.data(), y.data(), z.data(), h.data(), m.data(), numBodies, G, box, numShells, refAx.data(),
              refAy.data(), refAz.data(), refP.data());

    for (int i = 0; i < numBodies; ++i)
    {
        EXPECT_NEAR(h_ax[i], refAx[i], 1e-6);
        EXPECT_NEAR(h_ay[i], refAy[i], 1e-6);
        EXPECT_NEAR(h_az[i], refAz[i], 1e-6);
        EXPECT_NEAR(h_p[i], refP[i], 1e-6);
    }
}
