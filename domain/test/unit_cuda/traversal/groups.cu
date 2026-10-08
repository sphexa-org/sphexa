/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Cornerstone octree GPU testing
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#include "gtest/gtest.h"

#include <bitset>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "cstone/cuda/stream_holder.cuh"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/traversal/groups_gpu.cuh"
#include "cstone/traversal/groups_gpu.h"
#include "cstone/tree/cs_util.hpp"

using namespace cstone;

constexpr size_t targetSize = 64;
constexpr size_t nwt        = targetSize / GpuConfig::warpSize;
using SplitType             = util::array<GpuConfig::ThreadMask, nwt>;

TEST(TargetGroups, t0)
{
    LocalIndex groupSize = 8, first = 4, last = 34;

    StreamHolder stream;

    GroupData<execution::Gpu> groups;
    computeFixedGroups(stream.exec(), first, last, groupSize, groups);
    stream.sync();

    std::vector<LocalIndex> hgroups = toHost(groups.data);
    std::vector<LocalIndex> ref{4, 12, 20, 28, 34};
    EXPECT_EQ(hgroups, ref);
}

__device__ constexpr util::array<unsigned, 2> laneSeg(unsigned idx, unsigned warpSize_)
{
    return {idx % warpSize, idx / warpSize_};
}

//! @brief test input setup for findSplits
template<std::size_t N>
__global__ void findSplitTester(util::array<GpuConfig::ThreadMask, N>* splits)
{
    using T          = double;
    unsigned laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);

    util::array<Vec4<T>, N> pos;
    for (std::size_t k = 0; k < N; ++k)
    {
        T x    = T(laneIdx) + k * GpuConfig::warpSize;
        pos[k] = Vec4<T>{x, x, x, T(N * GpuConfig::warpSize)};
    }

    // introduce a split at position 0
    if (laneIdx == 0) { pos[0] = {-1, -1, -1, N * GpuConfig::warpSize}; }

    // introduce a split at position 2 due to interaction radius
    if (laneIdx == 2) { pos[0][3] = 0.99 * std::sqrt(3.); }

    // introduce a split at position 31
    if (laneIdx == 31)
    {
        pos[0][0] -= 0.5;
        pos[0][1] -= 0.5;
        pos[0][2] -= 0.5;
    }

    // introduce a split at position 33
    auto lk = laneSeg(33, GpuConfig::warpSize);
    if (lk[0] == laneIdx && lk[1] < N)
    {
        pos[lk[1]][0] -= 0.5;
        pos[lk[1]][1] -= 0.5;
        pos[lk[1]][2] -= 0.5;
    }

    *splits = findSplits(pos, T(3.01));
}

TEST(TargetGroups, findSplits)
{
    {
        thrust::device_vector<SplitType> d_splits(1);
        findSplitTester<<<1, GpuConfig::warpSize>>>(rawPtr(d_splits));
        SplitType split = d_splits[0];

        std::bitset<targetSize> splitBits;
        for (int k = nwt; k >= 0; --k)
        {
            splitBits <<= GpuConfig::warpSize;
            splitBits |= split[k];
        }

        EXPECT_EQ(splitBits.count(), 4);
        EXPECT_EQ(splitBits[0], 1);
        EXPECT_EQ(splitBits[2], 1);
        EXPECT_EQ(splitBits[31], 1);
        EXPECT_EQ(splitBits[33], 1);
    }
}

__global__ void makeSplitTester(SplitType splitMask, LocalIndex* splitLengths) { makeSplits(splitMask, splitLengths); }

TEST(TargetGroups, makeSplits)
{
    auto makeMask = [](auto a, auto b)
    {
        if constexpr (nwt == 2) // NOLINT
        {
            SplitType ret;
            ret[0] = a;
            ret[1] = b;
            return ret;
        }
        else { return SplitType{(uint64_t(b) << 32) + a}; } // NOLINT
    };

    {
        thrust::device_vector<LocalIndex> splitLengths(targetSize);
        SplitType splitMask = makeMask(0, 0);
        makeSplitTester<<<1, 1>>>(splitMask, rawPtr(splitLengths));
        EXPECT_EQ(splitLengths[0], 64);
    }
    {
        thrust::device_vector<LocalIndex> splitLengths(targetSize);
        SplitType splitMask = makeMask(1, 0);
        makeSplitTester<<<1, 1>>>(splitMask, rawPtr(splitLengths));
        EXPECT_EQ(splitLengths[0], 1);
        EXPECT_EQ(splitLengths[1], 63);
    }
    {
        thrust::device_vector<LocalIndex> splitLengths(targetSize);
        SplitType splitMask = makeMask(0, 1u << 30);
        makeSplitTester<<<1, 1>>>(splitMask, rawPtr(splitLengths));
        EXPECT_EQ(splitLengths[0], 63);
        EXPECT_EQ(splitLengths[1], 1);
    }
    {
        thrust::device_vector<LocalIndex> splitLengths(targetSize);
        SplitType splitMask = makeMask(2, 0);
        makeSplitTester<<<1, 1>>>(splitMask, rawPtr(splitLengths));
        EXPECT_EQ(splitLengths[0], 2);
        EXPECT_EQ(splitLengths[1], 62);
    }
    {
        thrust::device_vector<LocalIndex> splitLengths(targetSize);
        SplitType splitMask = makeMask(3, 0);
        makeSplitTester<<<1, 1>>>(splitMask, rawPtr(splitLengths));
        EXPECT_EQ(splitLengths[0], 1);
        EXPECT_EQ(splitLengths[1], 1);
        EXPECT_EQ(splitLengths[2], 62);
    }
    {
        thrust::device_vector<LocalIndex> splitLengths(targetSize);
        SplitType splitMask = makeMask(1u << 31, 1);

        makeSplitTester<<<1, 1>>>(splitMask, rawPtr(splitLengths));
        EXPECT_EQ(splitLengths[0], 32);
        EXPECT_EQ(splitLengths[1], 1);
        EXPECT_EQ(splitLengths[2], 31);
    }
    {
        thrust::device_vector<LocalIndex> splitLengths(targetSize);
        SplitType splitMask = makeMask(0, 8);

        makeSplitTester<<<1, 1>>>(splitMask, rawPtr(splitLengths));
        EXPECT_EQ(splitLengths[0], 36);
        EXPECT_EQ(splitLengths[1], 28);
    }
    {
        thrust::device_vector<LocalIndex> splitLengths(targetSize);
        SplitType splitMask = makeMask(0xFFFFFFFFu, 0x6FFFFFFFu);

        makeSplitTester<<<1, 1>>>(splitMask, rawPtr(splitLengths));
        for (std::size_t i = 0; i < targetSize - 1; ++i)
        {
            if (i == 60) { EXPECT_EQ(splitLengths[i], 2); }
            else { EXPECT_EQ(splitLengths[i], 1); }
        }
    }
    {
        thrust::device_vector<LocalIndex> splitLengths(targetSize);
        SplitType splitMask = makeMask(0xFFFFFFFF, 0x7FFFFFFF);

        makeSplitTester<<<1, 1>>>(splitMask, rawPtr(splitLengths));
        for (std::size_t i = 0; i < targetSize - 1; ++i)
        {
            EXPECT_EQ(splitLengths[i], 1);
        }
    }
}

/*! @brief group splitting test body, parametrized on the bounding box
 *
 * @param box       global bounding box, its x-extent has to be at least @a last so that the unit-spaced x
 *                  coordinates below fit inside
 * @param nodeEdge  edge length of the smallest leaf cell of any group, i.e. the cubic root of its volume
 *                  expressed as a fraction of the box volume. This is the quantity the kernel derives from
 *                  the leaf level and the per-axis SFC bit depths, and the reference value @p tolFactor is
 *                  calibrated against.
 *
 * The kernel measures distances and interaction radii in units of the geometric mean of the box edges. The
 * particles are placed along the box diagonal and the thresholds are expressed relative to their physical
 * spacing, such that all reference values below are independent of the box, except for @p nodeEdge.
 */
template<class T, class KeyType>
static void groupVolumesTest(const Box<T>& box, double nodeEdge)
{
    constexpr LocalIndex groupSize = 64;

    LocalIndex first        = 4;
    LocalIndex last         = 128;
    LocalIndex numParticles = last - first;
    LocalIndex numGroups    = iceil(numParticles, groupSize);

    // consecutive particles are placed along the box diagonal, a unit step in x keeps all of them inside the box
    T scaleY = box.ly() / box.lx();
    T scaleZ = box.lz() / box.lx();
    // physical distance between consecutive particles
    T stepLength = std::sqrt(1 + scaleY * scaleY + scaleZ * scaleZ);
    // distance between consecutive particles in the units of groupSplitsKernel
    double spacing = stepLength / std::cbrt(box.lx() * box.ly() * box.lz());

    auto leaves = OctreeMaker<KeyType>{}.divide().divide(2).makeTree();
    // nodeIdx                   0  1 |2  3  4  5  6   7  8  9 |10  11  12 13 14 15
    // fixed groups                |                 |                   |
    std::vector<unsigned> counts{4, 1, 8, 8, 8, 8, 31, 8, 8, 8, 16, 16, 16, 0, 0};
    std::vector<LocalIndex> layout(counts.size() + 1, 0);
    std::inclusive_scan(counts.begin(), counts.end(), layout.begin() + 1);

    // these coordinates do not lie in the leaf cells specified by layout, but this is irrelevant for this test case
    thrust::device_vector<T> x(last), y(last), z(last), h(last);
    thrust::sequence(x.begin(), x.end(), T(0), T(1));
    thrust::sequence(y.begin(), y.end(), T(0), scaleY);
    thrust::sequence(z.begin(), z.end(), T(0), scaleZ);
    thrust::fill(h.begin(), h.end(), box.maxExtent());
    // particle 6 in 2nd group get a smaller interaction radius, just enough to cause a split
    h[first + groupSize + 6] = 0.99 * stepLength / 2;
    // particle 7 in 2nd group's radius is just big enough not to cause a split
    h[first + groupSize + 7] = 1.01 * stepLength / 2;

    // introduce a split by increasing distance between particles 5 and 6
    x[5] -= 0.01;
    y[5] -= 0.01 * scaleY;
    z[5] -= 0.01 * scaleZ;

    thrust::device_vector<LocalIndex> groupDiv(numGroups);
    thrust::device_vector<SplitType> splitMasks(numGroups);

    thrust::device_vector<KeyType> d_leaves    = leaves;
    thrust::device_vector<LocalIndex> d_layout = layout;

    unsigned numThreads = 256;
    unsigned gridSize   = numGroups * GpuConfig::warpSize;
    {
        float tolFactor = spacing / nodeEdge * 1.01;
        groupSplitsKernel<groupSize, T><<<iceil(gridSize, numThreads), numThreads>>>(
            first, last, rawPtr(x), rawPtr(y), rawPtr(z), rawPtr(h), rawPtr(d_leaves), nNodes(leaves), rawPtr(d_layout),
            box, tolFactor, rawPtr(splitMasks), rawPtr(groupDiv), numGroups);

        thrust::host_vector<LocalIndex> h_groupDiv = groupDiv;
        thrust::host_vector<LocalIndex> ref        = std::vector<LocalIndex>{2, 2};
        EXPECT_EQ(h_groupDiv, ref);
    }
    {
        float tolFactor = spacing / nodeEdge * 0.99;
        groupSplitsKernel<groupSize, T><<<iceil(gridSize, numThreads), numThreads>>>(
            first, last, rawPtr(x), rawPtr(y), rawPtr(z), rawPtr(h), rawPtr(d_leaves), nNodes(leaves), rawPtr(d_layout),
            box, tolFactor, rawPtr(splitMasks), rawPtr(groupDiv), numGroups);

        thrust::host_vector<LocalIndex> h_groupDiv = groupDiv;
        thrust::host_vector<LocalIndex> ref        = std::vector<LocalIndex>{64, 60};
        EXPECT_EQ(h_groupDiv, ref);
    }

    {
        // Fixed size groups (4,68,128) get split into (4,6,68,75,128)
        //                            because of distance ^    ^ because of interaction radius
        DeviceVector<LocalIndex> temp, groups;

        StreamHolder stream;

        float tolFactor = spacing / nodeEdge * 1.01;
        computeGroupSplits(stream.exec(), first, last, rawPtr(x), rawPtr(y), rawPtr(z), rawPtr(h), rawPtr(d_leaves),
                           nNodes(leaves), rawPtr(d_layout), box, groupSize, tolFactor, temp, groups);
        stream.sync();

        std::vector<LocalIndex> h_groups = toHost(groups);
        std::vector<LocalIndex> ref{4, 6, 68, 75, 128};
        EXPECT_EQ(h_groups, ref);
    }
}

TEST(TargetGroups, groupVolumes)
{
    using T       = double;
    using KeyType = uint64_t;

    // cubic box, axesBits (21, 21, 21): the smallest leaf is a level-2 node covering 1/4 of each axis
    groupVolumesTest<T, KeyType>(Box<T>(0, 128), 1.0 / 4);
}

TEST(TargetGroups, groupVolumesMixD)
{
    using T       = double;
    using KeyType = uint64_t;

    // mixed-dimension box, axesBits (21, 20, 19): at level 2, only x has been subdivided twice, y once and z
    // not at all, so the smallest leaf covers (1/4, 1/2, 1) of the box, an edge length of cbrt(1/8) = 1/2
    groupVolumesTest<T, KeyType>(Box<T>(0, 128, 0, 64, 0, 32), 1.0 / 2);
}
