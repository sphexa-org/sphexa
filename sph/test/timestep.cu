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
    using T = double;
    // two consecutive groups of GroupView::groupSize particles: [0,64) and [64,128)
    sph::GroupView grpView{.firstBody = 0, .lastBody = 128};

    thrust::device_vector<T> divv(128);
    thrust::sequence(divv.begin(), divv.end(), 100);
    thrust::device_vector<float> groupDt(grpView.numGroups(), 1e10f);

    float Krho = 0.2;
    groupDivvTimestepGpu(Krho, grpView, rawPtr(divv), rawPtr(groupDt));

    thrust::host_vector<float> probe = groupDt;

    EXPECT_NEAR(probe[0], Krho / 163, 1e-10);
    EXPECT_NEAR(probe[1], Krho / 227, 1e-10);
}

TEST(TimestepGpu, Acc)
{
    using T = double;
    // two consecutive groups of GroupView::groupSize particles: [0,64) and [64,128)
    sph::GroupView grpView{.firstBody = 0, .lastBody = 128};

    thrust::device_vector<T> ax(128), ay(128), az(128), h(128);

    thrust::sequence(ax.begin(), ax.end(), 100);
    thrust::sequence(ay.begin(), ay.end(), 200);
    thrust::sequence(az.begin(), az.end(), 300);
    thrust::fill(h.begin(), h.begin() + 64, 1.0);
    thrust::fill(h.begin() + 64, h.end(), 2.0);

    thrust::device_vector<float> groupDt(grpView.numGroups(), 1e10f);

    float etaAcc = 0.2;
    groupAccTimestepGpu(etaAcc, grpView, rawPtr(ax), rawPtr(ay), rawPtr(az), rawPtr(h), rawPtr(groupDt));

    thrust::host_vector<float> probe = groupDt;

    EXPECT_NEAR(probe[0], etaAcc * std::sqrt((1.0) / std::sqrt(norm2(Vec3<T>{163, 263, 363}))), 1e-8);
    EXPECT_NEAR(probe[1], etaAcc * std::sqrt((2.0) / std::sqrt(norm2(Vec3<T>{227, 327, 427}))), 1e-8);
}
