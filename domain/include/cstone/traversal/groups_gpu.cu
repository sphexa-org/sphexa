/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Particle target grouping
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <stdexcept>

#include "cstone/cuda/device_vector.h"
#include "cstone/primitives/math.hpp"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/traversal/groups_gpu.cuh"

namespace cstone
{

//! @brief simple-fixed width group targets
__global__ static void
fixedGroupsKernel(LocalIndex first, LocalIndex last, unsigned groupSize, LocalIndex* groups, unsigned numGroups)
{
    LocalIndex tid = blockIdx.x * blockDim.x + threadIdx.x;

    LocalIndex groupIdx = tid / groupSize;
    if (groupIdx >= numGroups) { return; }

    if (tid == groupIdx * groupSize)
    {
        LocalIndex scan  = first + groupIdx * groupSize;
        groups[groupIdx] = scan;
        if (groupIdx + 1 == numGroups) { groups[numGroups] = last; }
    }
}

void computeFixedGroups(
    execution::Gpu exec, LocalIndex first, LocalIndex last, unsigned groupSize, GroupData<execution::Gpu>& groups)
{
    LocalIndex numBodies = last - first;
    LocalIndex numGroups = iceil(numBodies, groupSize);
    groups.data.resize(numGroups + 1);
    fixedGroupsKernel<<<iceil(numBodies, 256), 256, 0, exec>>>(first, last, groupSize, rawPtr(groups.data), numGroups);

    groups.firstBody  = first;
    groups.lastBody   = last;
    groups.numGroups  = numGroups;
    groups.groupStart = rawPtr(groups.data);
    groups.groupEnd   = rawPtr(groups.data) + 1;
}

//! @brief convenience wrapper for groupSplitsKernel
template<unsigned groupSize, class Tc, class T, class KeyType>
void computeGroupSplitsImpl(
    execution::Gpu exec,
    LocalIndex first,
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
    DeviceVector<util::array<GpuConfig::ThreadMask, groupSize / GpuConfig::warpSize>>& splitMasks,
    DeviceVector<LocalIndex>& numSplitsPerGroup,
    DeviceVector<LocalIndex>& groups)
{
    LocalIndex numParticles   = last - first;
    LocalIndex numFixedGroups = iceil(numParticles, groupSize);
    unsigned numThreads       = 256;
    unsigned gridSize         = numFixedGroups * GpuConfig::warpSize;

    splitMasks.resize(numFixedGroups);

    numSplitsPerGroup.reserve(numFixedGroups * 1.1);
    numSplitsPerGroup.resize(numFixedGroups + 1);

    if (numFixedGroups > 0)
        groupSplitsKernel<groupSize><<<iceil(gridSize, numThreads), numThreads, 0, exec>>>(
            first, last, x, y, z, h, leaves, numLeaves, layout, box, tolFactor, rawPtr(splitMasks),
            rawPtr(numSplitsPerGroup), numFixedGroups);

    groups.reserve(numFixedGroups * 1.1);
    groups.resize(numFixedGroups + 1);
    exclusiveScan(exec, rawPtr(numSplitsPerGroup), rawPtr(numSplitsPerGroup) + numFixedGroups + 1, rawPtr(groups));
    LocalIndex newNumGroups;
    memcpyD2HAsync(exec, rawPtr(groups) + groups.size() - 1, 1, &newNumGroups);
    syncGpu(exec);

    auto& newGroupSizes = numSplitsPerGroup;
    newGroupSizes.resize(newNumGroups + 1);

    if (numFixedGroups > 0)
        makeSplitsKernel<<<numFixedGroups, numThreads, 0, exec>>>(rawPtr(splitMasks), rawPtr(groups), numFixedGroups,
                                                                  rawPtr(newGroupSizes));

    groups.resize(newNumGroups + 1);
    exclusiveScan(exec, rawPtr(newGroupSizes), rawPtr(newGroupSizes) + newNumGroups + 1, rawPtr(groups), first);
    memcpyH2DAsync(exec, &last, 1, rawPtr(groups) + groups.size() - 1);
}

template<class Tc, class T, class KeyType>
void computeGroupSplits(execution::Gpu exec,
                        LocalIndex first,
                        LocalIndex last,
                        const Tc* x,
                        const Tc* y,
                        const Tc* z,
                        const T* h,
                        const KeyType* leaves,
                        TreeNodeIndex numLeaves,
                        const LocalIndex* layout,
                        const Box<Tc> box,
                        unsigned groupSize,
                        float tolFactor,
                        DeviceVector<LocalIndex>& numSplitsPerGroup,
                        DeviceVector<LocalIndex>& groups)
{
    if (groupSize == GpuConfig::warpSize)
    {
        DeviceVector<util::array<GpuConfig::ThreadMask, 1>> splitMasks;
        computeGroupSplitsImpl<GpuConfig::warpSize>(exec, first, last, x, y, z, h, leaves, numLeaves, layout, box,
                                                    tolFactor, splitMasks, numSplitsPerGroup, groups);
    }
    else if (groupSize == 2 * GpuConfig::warpSize)
    {
        DeviceVector<util::array<GpuConfig::ThreadMask, 2>> splitMasks;
        computeGroupSplitsImpl<2 * GpuConfig::warpSize>(exec, first, last, x, y, z, h, leaves, numLeaves, layout, box,
                                                        tolFactor, splitMasks, numSplitsPerGroup, groups);
    }
    else { throw std::runtime_error("Unsupported spatial group size\n"); }
}

#define COMPUTE_GROUP_SPLITS(Tc, T, KeyType)                                                                           \
    template void computeGroupSplits(execution::Gpu, LocalIndex first, LocalIndex last, const Tc* x, const Tc* y,      \
                                     const Tc* z, const T* h, const KeyType* leaves, TreeNodeIndex numLeaves,          \
                                     const LocalIndex* layout, const Box<Tc> box, unsigned groupSize, float tolFactor, \
                                     DeviceVector<LocalIndex>& numSplitsPerGroup, DeviceVector<LocalIndex>& groups)

COMPUTE_GROUP_SPLITS(double, double, uint64_t);
COMPUTE_GROUP_SPLITS(double, float, uint64_t);
COMPUTE_GROUP_SPLITS(float, float, uint64_t);

} // namespace cstone
