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
#include <climits>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "cstone/cuda/thrust_util.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/traversal/groups.hpp"
#include "cstone/traversal/groups_gpu.cuh"
#include "cstone/tree/cs_util.hpp"

using namespace cstone;

constexpr size_t targetSize = 64;
constexpr size_t nwt        = targetSize / GpuConfig::warpSize;
using SplitType             = util::array<GpuConfig::ThreadMask, nwt>;

//! @brief test-local copy of the removed production standalone split pipeline

/*! @brief Count number of zero bits between 1-bits for the bitstream defined by @p splits
 *
 * @tparam N                  number of particles per thread
 * @param[in]  split          the split-bitmask
 * @param[out] splitLengths   output number of 0-bits between 1-bits, with space for popCount(split) + 1 elements
 *
 * Examples, assuming split is a 64-bit stream (either 2x32bit or 1x64bit):
 *    split = 0x0 --> splitLengths = {64}
 *    split = 0x1 --> splitLengths = {1, 63}
 *    split = 0x2 --> splitLengths = {2, 62}
 *    split = 0x3 --> splitLengths = {1, 1, 61}
 */
template<std::size_t N>
__device__ void makeSplits(util::array<GpuConfig::ThreadMask, N> split, LocalIndex* splitLengths)
{
    auto readSegment = [&splitLengths](GpuConfig::ThreadMask mask, int carry)
    {
        int bitsRemaining = CHAR_BIT * sizeof(GpuConfig::ThreadMask);
        while (popCount(mask))
        {
            int length = countTrailingZeros(mask) + 1;
            bitsRemaining -= length;
            *splitLengths++ = length + carry;
            carry           = 0;
            mask >>= length;
        }
        return carry + bitsRemaining;
    };

    int carry = 0;
    for (std::size_t k = 0; k < N; ++k)
    {
        carry = readSegment(split[k], carry);
    }
    *splitLengths = carry;
}

template<size_t N>
__global__ void makeSplitsKernel(const util::array<GpuConfig::ThreadMask, N>* splitMasks,
                                 const LocalIndex* groupOffsets,
                                 LocalIndex numFixedGroups,
                                 LocalIndex* splitLengths)
{
    LocalIndex tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numFixedGroups) { return; }

    makeSplits(splitMasks[tid], splitLengths + groupOffsets[tid]);
}

/*! @brief compute split pattern and split counts for each initial fixed-size target particle group
 *
 * @tparam groupSize           a multiple of GpuConfig::warpSize
 * @tparam T                   float or double
 * @tparam KeyType             32- or 64-bit unsigned integer
 * @param[in]  first           first particle from x,y,z,h to consider
 * @param[in]  last            last particle to consider
 * @param[in]  x               local x particle coordinates
 * @param[in]  y               local y particle coordinates
 * @param[in]  z               local z particle coordinates
 * @param[in]  h               local smoothing lengths
 * @param[in]  leaves          cornerstone leaf-cell array
 * @param[in]  numLeaves       number of leaves in @p leaves
 * @param[in]  layout          layout[i] is the x,y,z,h-array particle index of the first particle in leaf i,
 * @param[in]  box             global coordinate bounding box
 * @param[in]  tolFactor       max distance between consecutive particles is
 *                             tolFactor * cbrt(smallest leaf node size in group)
 * @param[out] splitMasks      split mask for each of the ceil((last-first)/groupSize) fixed-size groups
 * @param[out] numSplitsPerGroup 1 + number-of-1-bits in @p splitMasks for each fixed-size group
 */
template<unsigned groupSize, class Tc, class T, class KeyType>
__global__ void groupSplitsKernel(LocalIndex first,
                                  LocalIndex last,
                                  const Tc* x,
                                  const Tc* y,
                                  const Tc* z,
                                  const T* h,
                                  const KeyType* leaves,
                                  TreeNodeIndex numLeaves,
                                  const LocalIndex* layout,
                                  const Box<Tc> box,
                                  float tolFactor,
                                  util::array<GpuConfig::ThreadMask, groupSize / GpuConfig::warpSize>* splitMasks,
                                  LocalIndex* numSplitsPerGroup,
                                  unsigned /*numGroups*/)
{
    constexpr LocalIndex nwt = groupSize / GpuConfig::warpSize;

    LocalIndex tid     = blockIdx.x * blockDim.x + threadIdx.x;
    LocalIndex warpIdx = tid / GpuConfig::warpSize;
    LocalIndex laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);

    if (warpIdx * groupSize >= last - first) { return; }

    auto splitMask = evalGroupSplitMask<groupSize, Tc, T, KeyType>(first + warpIdx * groupSize, last, x, y, z, h,
                                                                   leaves, numLeaves, layout, box, tolFactor);

    if (laneIdx == 0)
    {
        splitMasks[warpIdx] = splitMask;
        int numSubGroups    = 1;
        for (LocalIndex k = 0; k < nwt; ++k)
        {
            numSubGroups += popCount(splitMask[k]);
        }
        numSplitsPerGroup[warpIdx] = numSubGroups;
    }
}

//! @brief basic test-local replacement for the removed production computeGroupSplits
//!
//! Single fixed group size of targetSize (no warpSize dispatch), no growth heuristics, default stream.
template<class Tc, class T, class KeyType>
thrust::host_vector<LocalIndex> computeGroupSplits(LocalIndex first, LocalIndex last, const Tc* x, const Tc* y,
                                                   const Tc* z, const T* h, const KeyType* leaves,
                                                   TreeNodeIndex numLeaves, const LocalIndex* layout,
                                                   const Box<Tc> box, float tolFactor)
{
    LocalIndex numFixedGroups = iceil(last - first, targetSize);

    thrust::device_vector<SplitType> splitMasks(numFixedGroups);
    thrust::device_vector<LocalIndex> numSplitsPerGroup(numFixedGroups + 1);

    unsigned numThreads = 256;
    unsigned gridSize   = numFixedGroups * GpuConfig::warpSize;
    if (numFixedGroups > 0)
    {
        groupSplitsKernel<targetSize, Tc, T, KeyType><<<iceil(gridSize, numThreads), numThreads>>>(
            first, last, x, y, z, h, leaves, numLeaves, layout, box, tolFactor, rawPtr(splitMasks),
            rawPtr(numSplitsPerGroup), numFixedGroups);
    }

    thrust::device_vector<LocalIndex> groupOffsets(numFixedGroups + 1);
    thrust::exclusive_scan(numSplitsPerGroup.begin(), numSplitsPerGroup.end(), groupOffsets.begin());
    LocalIndex newNumGroups = groupOffsets.back(); // proxy read syncs

    thrust::device_vector<LocalIndex> groupSizes(newNumGroups);
    if (numFixedGroups > 0)
    {
        makeSplitsKernel<<<numFixedGroups, numThreads>>>(rawPtr(splitMasks), rawPtr(groupOffsets), numFixedGroups,
                                                         rawPtr(groupSizes));
    }

    thrust::device_vector<LocalIndex> groups(newNumGroups + 1);
    thrust::exclusive_scan(groupSizes.begin(), groupSizes.begin() + newNumGroups, groups.begin(), first);
    groups[newNumGroups] = last;

    return thrust::host_vector<LocalIndex>(groups);
}

TEST(TargetGroups, t0)
{
    //! all groups active without an index list: group ranges derive from the body range
    GroupView groups{.firstBody = 4, .lastBody = 260};

    EXPECT_EQ(groups.numGroups(), 4);
    EXPECT_EQ(groups.groupStart(0), 4);
    EXPECT_EQ(groups.groupEnd(0), 68);
    EXPECT_EQ(groups.groupStart(3), 196);
    EXPECT_EQ(groups.groupEnd(3), 260);
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

    *splits = findSplits(pos, 3.01);
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

TEST(TargetGroups, groupVolumes)
{
    using T                        = double;
    using KeyType                  = uint64_t;
    constexpr LocalIndex groupSize = 64;

    LocalIndex first        = 4;
    LocalIndex last         = 128;
    LocalIndex numParticles = last - first;
    LocalIndex numGroups    = iceil(numParticles, groupSize);

    Box<T> box(0, last);
    double distCrit = std::cbrt(box.lx() * box.ly() * box.lz() / 64);

    auto leaves = OctreeMaker<KeyType>{}.divide().divide(2).makeTree();
    // nodeIdx                   0  1 |2  3  4  5  6   7  8  9 |10  11  12 13 14 15
    // fixed groups                |                 |                   |
    std::vector<unsigned> counts{4, 1, 8, 8, 8, 8, 31, 8, 8, 8, 16, 16, 16, 0, 0};
    std::vector<LocalIndex> layout(counts.size() + 1, 0);
    std::inclusive_scan(counts.begin(), counts.end(), layout.begin() + 1);

    // these coordinates do not lie in the leaf cells specified by layout, but this is irrelevant for this test case
    thrust::device_vector<T> x(last), y(last), z(last), h(last);
    thrust::sequence(x.begin(), x.end(), 0);
    thrust::sequence(y.begin(), y.end(), 0);
    thrust::sequence(z.begin(), z.end(), 0);
    thrust::fill(h.begin(), h.end(), box.maxExtent());
    // particle 6 in 2nd group get a smaller interaction radius, just enough to cause a split
    h[first + groupSize + 6] = 0.99 * std::sqrt(3.) / 2;
    // particle 7 in 2nd group's radius is just big enough not to cause a split
    h[first + groupSize + 7] = 1.01 * std::sqrt(3.) / 2;

    // introduce a split by increasing distance between particles 5 and 6
    x[5] -= 0.01;
    y[5] -= 0.01;
    z[5] -= 0.01;

    thrust::device_vector<LocalIndex> groupDiv(numGroups);
    thrust::device_vector<SplitType> splitMasks(numGroups);

    thrust::device_vector<KeyType> d_leaves    = leaves;
    thrust::device_vector<LocalIndex> d_layout = layout;

    unsigned numThreads = 256;
    unsigned gridSize   = numGroups * GpuConfig::warpSize;
    {
        float tolFactor = std::sqrt(3.0) / distCrit * 1.01;
        groupSplitsKernel<groupSize, T><<<iceil(gridSize, numThreads), numThreads>>>(
            first, last, rawPtr(x), rawPtr(y), rawPtr(z), rawPtr(h), rawPtr(d_leaves), nNodes(leaves), rawPtr(d_layout),
            box, tolFactor, rawPtr(splitMasks), rawPtr(groupDiv), numGroups);

        thrust::host_vector<LocalIndex> h_groupDiv = groupDiv;
        thrust::host_vector<LocalIndex> ref        = std::vector<LocalIndex>{2, 2};
        EXPECT_EQ(h_groupDiv, ref);
    }
    {
        float tolFactor = std::sqrt(3.0) / distCrit * 0.99;
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
        float tolFactor = std::sqrt(3.0) / distCrit * 1.01;
        thrust::host_vector<LocalIndex> h_groups =
            computeGroupSplits(first, last, rawPtr(x), rawPtr(y), rawPtr(z), rawPtr(h), rawPtr(d_leaves),
                               nNodes(leaves), rawPtr(d_layout), box, tolFactor);
        thrust::host_vector<LocalIndex> ref = std::vector<LocalIndex>{4, 6, 68, 75, 128};
        EXPECT_EQ(h_groups, ref);
    }
}
