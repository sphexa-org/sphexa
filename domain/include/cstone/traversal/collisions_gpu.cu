/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  GPU driver for halo discovery using traversal of an octree
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/primitives/math.hpp"
#include "cstone/traversal/collisions_gpu.h"
#include "cstone/traversal/macs.hpp"

namespace cstone
{

template<class KeyType, class T>
__global__ void findHalosKernel(const KeyType* nodePrefixes,
                                const TreeNodeIndex* childOffsets,
                                const TreeNodeIndex* parents,
                                const Vec3<T>* nodeCenters,
                                const Vec3<T>* nodeSizes,
                                const KeyType* leaves,
                                const Vec3<T>* searchCenters,
                                const Vec3<T>* searchSizes,
                                const Box<T> box,
                                TreeNodeIndex firstNode,
                                TreeNodeIndex lastNode,
                                uint8_t* collisionFlags)
{
    unsigned leafIdx = blockIdx.x * blockDim.x + threadIdx.x + firstNode;

    if (leafIdx < lastNode)
    {
        Vec3<T> tC         = searchCenters[leafIdx];
        Vec3<T> tS         = searchSizes[leafIdx];
        KeyType lowestKey  = leaves[firstNode];
        KeyType highestKey = leaves[lastNode];

        // if the halo box is fully inside the assigned SFC range, we skip collision detection
        if (containedIn(lowestKey, highestKey, tC, tS, box)) { return; }

        // mark all colliding node indices outside [lowestKey:highestKey]
        findCollisions(nodePrefixes, childOffsets, parents, nodeCenters, nodeSizes, tC, tS, box, lowestKey, highestKey,
                       collisionFlags);
    }
}

template<class KeyType, class T>
void findHalosGpu(const KeyType* prefixes,
                  const TreeNodeIndex* childOffsets,
                  const TreeNodeIndex* parents,
                  const Vec3<T>* nodeCenters,
                  const Vec3<T>* nodeSizes,
                  const KeyType* leaves,
                  const Vec3<T>* searchCenters,
                  const Vec3<T>* searchSizes,
                  const Box<T>& box,
                  TreeNodeIndex firstNode,
                  TreeNodeIndex lastNode,
                  uint8_t* collisionFlags)
{
    constexpr unsigned numThreads = 128;
    unsigned numBlocks            = iceil(lastNode - firstNode, numThreads);

    if (numBlocks == 0) { return; }
    findHalosKernel<<<numBlocks, numThreads>>>(prefixes, childOffsets, parents, nodeCenters, nodeSizes, leaves,
                                               searchCenters, searchSizes, box, firstNode, lastNode, collisionFlags);
}

#define FIND_HALOS_GPU(KeyType, T)                                                                                     \
    template void findHalosGpu(const KeyType* prefixes, const TreeNodeIndex* childOffsets,                             \
                               const TreeNodeIndex* parents, const Vec3<T>* nodeCenters, const Vec3<T>* nodeSizes,     \
                               const KeyType* leaves, const Vec3<T>* searchCenters, const Vec3<T>* searchSizes,        \
                               const Box<T>& box, TreeNodeIndex firstNode, TreeNodeIndex lastNode,                     \
                               uint8_t* collisionFlags)

FIND_HALOS_GPU(uint32_t, float);
FIND_HALOS_GPU(uint64_t, float);
FIND_HALOS_GPU(uint64_t, double);

template<class T, class KeyType>
__global__ void markMacsGpuKernel(const KeyType* prefixes,
                                  const TreeNodeIndex* childOffsets,
                                  const TreeNodeIndex* parents,
                                  const Vec4<T>* centers,
                                  const Box<T> box,
                                  const KeyType* focusNodes,
                                  TreeNodeIndex numFocusNodes,
                                  bool limitSource,
                                  uint8_t* markings)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= numFocusNodes) { return; }

    KeyType focusStart = focusNodes[0];
    KeyType focusEnd   = focusNodes[numFocusNodes];

    IBox target    = sfcIBox(sfcKey(focusNodes[tid]), sfcKey(focusNodes[tid + 1]));
    IBox targetExt = IBox(target.xmin() - 1, target.xmax() + 1, target.ymin() - 1, target.ymax() + 1, target.zmin() - 1,
                          target.zmax() + 1);
    if (containedIn(focusStart, focusEnd, targetExt)) { return; }

    auto [targetCenter, targetSize] = centerAndSize<KeyType>(target, box);
    unsigned maxLevel               = maxTreeLevel<KeyType>{};
    if (limitSource) { maxLevel = stl::max(int(treeLevel(focusNodes[tid + 1] - focusNodes[tid])) - 1, 0); }
    markMacPerBox(targetCenter, targetSize, maxLevel, prefixes, childOffsets, parents, centers, box, focusStart,
                  focusEnd, markings);
}

template<class T, class KeyType>
void markMacsGpu(const KeyType* prefixes,
                 const TreeNodeIndex* childOffsets,
                 const TreeNodeIndex* parents,
                 const Vec4<T>* centers,
                 const Box<T>& box,
                 const KeyType* focusNodes,
                 TreeNodeIndex numFocusNodes,
                 bool limitSource,
                 uint8_t* markings)
{
    constexpr unsigned numThreads = 128;
    unsigned numBlocks            = iceil(numFocusNodes, numThreads);

    if (numFocusNodes)
    {
        markMacsGpuKernel<<<numBlocks, numThreads>>>(prefixes, childOffsets, parents, centers, box, focusNodes,
                                                     numFocusNodes, limitSource, markings);
    }
}

#define MARK_MACS_GPU(KeyType, T)                                                                                      \
    template void markMacsGpu(const KeyType* prefixes, const TreeNodeIndex* childOffsets,                              \
                              const TreeNodeIndex* parents, const Vec4<T>* centers, const Box<T>& box,                 \
                              const KeyType* focusNodes, TreeNodeIndex numFocusNodes, bool limitSource,                \
                              uint8_t* markings)

MARK_MACS_GPU(uint64_t, double);
MARK_MACS_GPU(uint64_t, float);
MARK_MACS_GPU(unsigned, float);

} // namespace cstone
