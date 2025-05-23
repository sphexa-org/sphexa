
/*! @file
 * @brief time integration tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <cmath>
#include <iostream>
#include "gtest/gtest.h"
#include "sph/positions.hpp"

using namespace cstone;
using namespace sph;

TEST(Integrator, timeReversal)
{
    using T = float;
    double dtn = 0.1, dtnm1 = 0.5;

    Box<float> box(-100, 100);

    /*!     dtnm1            dtn
     * n-1               n         n+1
     * |-----------------|---------|
     *
     */
    const Vec3<T> Xn{1, 1, 1}, dXn{0.1, 0.1, 0.1}, An{2, 2, 2};

    Vec3<T> Xnp1, Vnp1, dXnp1;
    std::tie(Xnp1, Vnp1, dXnp1) = positionUpdate(dtn, dtnm1, Xn, An, dXn, box);

    // advance to an intermediate time
    Vec3<T> Xtmp, Vtmp;
    std::tie(Xtmp, Vtmp, std::ignore) = positionUpdate(0.5 * dtn, dtnm1, Xn, An, dXn, box);

    // undo last advance to an intermediate time
    Vec3<T> Xn_re;
    std::tie(Xn_re, std::ignore, std::ignore) = positionUpdate(-0.5 * dtn, dtnm1, Xtmp, An, dXn, box);

    // advance to final time
    Vec3<T> Xnp1_ts, Vnp1_ts, dXnp1_ts;
    std::tie(Xnp1_ts, Vnp1_ts, dXnp1_ts) = positionUpdate(dtn, dtnm1, Xn_re, An, dXn, box);

    EXPECT_EQ(Xnp1, Xnp1_ts);
    EXPECT_EQ(Vnp1, Vnp1_ts);
    EXPECT_EQ(dXnp1, dXnp1_ts);
}

TEST(Integrator, timeEnergyReversal)
{
    using T = float;

    T dtn = 0.1, dtnm1 = 0.2;
    T dU = 2.0, dUm1 = 3.0;

    auto Unp1 = energyUpdate(10.0, dtn, dtnm1, dU, dUm1);
    EXPECT_NEAR(Unp1, 10.175, 1e-7);

    auto Un = energyUpdate(10.175, -dtn, dtnm1, dU, dUm1);
    EXPECT_NEAR(Un, 10.0, 1e-7);
}
