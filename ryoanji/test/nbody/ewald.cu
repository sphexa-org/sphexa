/*! @file
 * @brief Compare the Ewald GPU kernel against the CPU version
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/focus/source_center.hpp"
#include "cstone/traversal/groups_gpu.cu"

#include "dataset.hpp"
#include "ryoanji/nbody/cartesian_qpole.hpp"
#include "ryoanji/nbody/ewald.hpp"
#include "ryoanji/interface/ewald.cuh"

using namespace cstone;
using namespace ryoanji;

TEST(Ewald, MatchCpu)
{
    using T                = double;
    using KeyType          = uint64_t;
    using MultipoleType    = CartesianQuadrupole<float>;
    using SourceCenterType = util::array<T, 4>;

    /// Test input ************************
    size_t         numBodies = 100;
    T              G         = 1.5;
    T              coordMax  = 3.0;
    cstone::Box<T> box(-coordMax, coordMax, cstone::BoundaryType::periodic);
    EwaldSettings  settings{.numReplicaShells = 1, .lCut = 2.6, .hCut = 2.8, .alpha_scale = 2.0};

    std::vector<T> x(numBodies), y(numBodies), z(numBodies), m(numBodies), h(numBodies);
    ryoanji::makeCubeBodies(x.data(), y.data(), z.data(), m.data(), h.data(), numBodies, coordMax);
    ///**************************************

    MultipoleType rootMultipole;
    auto          centerMass = massCenter<T>(x.data(), y.data(), z.data(), m.data(), 0, numBodies);
    P2M(x.data(), y.data(), z.data(), m.data(), 0, numBodies, centerMass, rootMultipole);

    // upload to device
    thrust::device_vector<T> d_x = x, d_y = y, d_z = z, d_m = m, d_h = h;
    thrust::device_vector<T> p(numBodies), ax(numBodies), ay(numBodies), az(numBodies);

    GroupData<GpuTag> groups;
    computeFixedGroups(0, numBodies, GpuConfig::warpSize, groups);

    T utot = 0;
    computeGravityEwaldGpu(makeVec3(centerMass), rootMultipole, groups.view(), rawPtr(d_x), rawPtr(d_y), rawPtr(d_z),
                           rawPtr(d_m), box, G, rawPtr(p), rawPtr(ax), rawPtr(ay), rawPtr(az), &utot, settings);

    T              utotRef = 0;
    std::vector<T> refP(numBodies), refAx(numBodies), refAy(numBodies), refAz(numBodies);
    computeGravityEwald(makeVec3(centerMass), rootMultipole, 0, numBodies, x.data(), y.data(), z.data(), m.data(), box,
                        G, refP.data(), refAx.data(), refAy.data(), refAz.data(), &utotRef, settings);

    // download body accelerations
    thrust::host_vector<T> h_p = p, h_ax = ax, h_ay = ay, h_az = az;

    EXPECT_NEAR(utotRef, utot, 1e-7);

    for (int i = 0; i < numBodies; ++i)
    {
        EXPECT_NEAR(h_p[i], refP[i], 1e-6);
        EXPECT_NEAR(h_ax[i], refAx[i], 1e-6);
        EXPECT_NEAR(h_ay[i], refAy[i], 1e-6);
        EXPECT_NEAR(h_az[i], refAz[i], 1e-6);
    }
}
