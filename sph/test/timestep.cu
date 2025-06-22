/*! @file
 * @brief  Tests for block time-steps related GPU device kernels
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <iostream>

#include "gtest/gtest.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "cstone/cuda/thrust_util.cuh"
#include "sph/sph_gpu.hpp"

using namespace cstone;
using namespace sph;

TEST(TimestepGpu, Divv)
{
    using T                                  = double;
    thrust::device_vector<LocalIndex> groups = std::vector<int>{10, 20, 40};
    GroupView                         grpView{10, 40, 2, rawPtr(groups), rawPtr(groups) + 1};

    thrust::device_vector<T> divv(40);
    thrust::sequence(divv.begin(), divv.end(), 100);
    thrust::device_vector<float> groupDt(groups.size() - 1, 1e10f);

    float Krho = 0.2;
    groupDivvTimestepGpu(Krho, grpView, rawPtr(divv), rawPtr(groupDt));

    thrust::host_vector<float> probe = groupDt;

    EXPECT_NEAR(probe[0], Krho / 119, 1e-10);
    EXPECT_NEAR(probe[1], Krho / 139, 1e-10);
}

TEST(TimestepGpu, Acc)
{
    using T                                  = double;
    thrust::device_vector<LocalIndex> groups = std::vector<int>{10, 20, 40};
    GroupView                         grpView{10, 40, 2, rawPtr(groups), rawPtr(groups) + 1};

    thrust::device_vector<T> ax(40), ay(40), az(40);

    thrust::sequence(ax.begin(), ax.end(), 100);
    thrust::sequence(ay.begin(), ay.end(), 200);
    thrust::sequence(az.begin(), az.end(), 300);

    thrust::device_vector<float> groupDt(groups.size() - 1, 1e10f);

    float etaAcc = 0.2;
    groupAccTimestepGpu(etaAcc, grpView, rawPtr(ax), rawPtr(ay), rawPtr(az), rawPtr(groupDt));

    thrust::host_vector<float> probe = groupDt;

    EXPECT_NEAR(probe[0], etaAcc / std::sqrt(std::sqrt(norm2(Vec3<T>{100 + 19, 200 + 19, 300 + 19}))), 1e-9);
    EXPECT_NEAR(probe[1], etaAcc / std::sqrt(std::sqrt(norm2(Vec3<T>{100 + 39, 200 + 39, 300 + 39}))), 1e-9);
}
