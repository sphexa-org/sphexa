/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Neighbor search on GPU with breadth-first warp-aware octree traversal
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/cuda/gpu_config.cuh"
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/primitives/warpscan.cuh"
#include "cstone/sfc/box.hpp"
#include "cstone/findneighbors.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/tree/octree.hpp"

namespace cstone
{

struct TravConfig
{
    //! @brief size of global workspace memory per warp, must be a power of 2
    static constexpr unsigned memPerWarp = 256 * GpuConfig::warpSize;
    static_assert((memPerWarp & (memPerWarp - 1)) == 0);

    //! @brief number of threads per block for the traversal kernel
    static constexpr unsigned numThreads = 128;

    static constexpr unsigned numWarpsPerSm = 40;
    //! @brief maximum number of simultaneously active blocks
    inline static unsigned maxNumActiveBlocks =
        GpuConfig::smCount * (numWarpsPerSm / (numThreads / GpuConfig::warpSize));

    //! @brief number of particles per target, i.e. per warp
    static constexpr unsigned targetSize = GpuConfig::warpSize;

    //! @brief number of warps per target, used all over the place, hence the short name
    static constexpr unsigned nwt = targetSize / GpuConfig::warpSize;

    //! @brief number of blocks to launch, no longer adapts to grids that are too small to saturate all SMs
    static unsigned numBlocks() { return TravConfig::maxNumActiveBlocks; }

    //! @brief compute storage needed for traversal stack
    static unsigned poolSize()
    {
        unsigned numWarpsPerBlock = TravConfig::numThreads / GpuConfig::warpSize;
        return TravConfig::memPerWarp * numWarpsPerBlock * maxNumActiveBlocks;
    }
};

__device__ __forceinline__ int ringAddr(const int i) { return i & (TravConfig::memPerWarp - 1); }

/*! @brief apply neighbor particle interaction within a cutoff
 *
 * @tparam       T                 float or double
 * @param[in]    sourceBody        source body x,y,z
 * @param[in]    validLaneMask     number of lanes that contain valid source bodies
 * @param[in]    bodyIdx_i         target body index
 * @param[in]    pos_i             target body x,y,z,h
 * @param[in]    box               global coordinate bounding box
 * @param[in]    sourceBodyIdx     index of source body of each lane
 * @param[in]    handleInteraction interaction-handling functor
 *
 * Number of computed particle-particle pairs per call is GpuConfig::warpSize^2 * TravConfig::nwt
 */
template<bool UsePbc, class Tc, class InteractionsHandler>
__device__ void applyInteraction(Vec3<Tc> sourceBody,
                                 int numLanesValid,
                                 const util::array<Vec4<Tc>, TravConfig::nwt>& pos_i,
                                 const Box<Tc>& box,
                                 cstone::LocalIndex sourceBodyIdx,
                                 const InteractionsHandler& handleInteraction)
{
    for (int j = 0; j < numLanesValid; j++)
    {
        Vec3<Tc> pos_j{shflSync(sourceBody[0], j), shflSync(sourceBody[1], j), shflSync(sourceBody[2], j)};
        cstone::LocalIndex idx_j = shflSync(sourceBodyIdx, j);

#pragma unroll
        for (unsigned k = 0; k < TravConfig::nwt; k++)
        {
            Tc d2 = distanceSq<UsePbc>(pos_j[0], pos_j[1], pos_j[2], pos_i[k][0], pos_i[k][1], pos_i[k][2], box);
            if (d2 < pos_i[k][3] && d2 > Tc(0.0)) { handleInteraction(k, idx_j); }
        }
    }
}

template<bool UsePbc, class T, std::enable_if_t<UsePbc, int> = 0>
__device__ __forceinline__ bool cellOverlap(const Vec3<T>& curSrcCenter,
                                            const Vec3<T>& curSrcSize,
                                            const Vec3<T>& targetCenter,
                                            const Vec3<T>& targetSize,
                                            const Box<T>& box)
{
    return norm2(minDistance(curSrcCenter, curSrcSize, targetCenter, targetSize, box)) == T(0.0);
}

template<bool UsePbc, class T, std::enable_if_t<!UsePbc, int> = 0>
__device__ __forceinline__ bool cellOverlap(const Vec3<T>& curSrcCenter,
                                            const Vec3<T>& curSrcSize,
                                            const Vec3<T>& targetCenter,
                                            const Vec3<T>& targetSize,
                                            const Box<T>& /*box*/)
{
    return norm2(minDistance(curSrcCenter, curSrcSize, targetCenter, targetSize)) == T(0.0);
}

template<class Tc>
__device__ __forceinline__ bool tightOverlap(int laneIdx,
                                             bool isClose,
                                             const Vec3<Tc>& srcCenter,
                                             const Vec3<Tc>& srcSize,
                                             const util::array<Vec4<Tc>, TravConfig::nwt>& pos_i,
                                             const cstone::Box<Tc>& box)
{
    GpuConfig::ThreadMask closeLanes = ballotSync(isClose);

    bool isTightClose = isClose;
    for (int lane = 0; lane < GpuConfig::warpSize; ++lane)
    {
        // skip if this lane does not have a close source
        if (!((GpuConfig::ThreadMask(1) << lane) & closeLanes)) { continue; }

        // broadcast srcCenter/size of this lane
        Vec3<Tc> center{shflSync(srcCenter[0], lane), shflSync(srcCenter[1], lane), shflSync(srcCenter[2], lane)};
        Vec3<Tc> size{shflSync(srcSize[0], lane), shflSync(srcSize[1], lane), shflSync(srcSize[2], lane)};

        // does any of the individual target particles overlap with center/size ?
        bool overlapsWithLaneParticle = false;
        for (unsigned k = 0; k < TravConfig::nwt; ++k)
        {
            overlapsWithLaneParticle |= norm2(minDistance(makeVec3(pos_i[k]), center, size, box)) < pos_i[k][3];
        }
        GpuConfig::ThreadMask anyOverlaps = ballotSync(overlapsWithLaneParticle);
        if (lane == laneIdx) { isTightClose = anyOverlaps; }
    }
    return isTightClose;
}

/*! @brief traverse one warp with up to TravConfig::targetSize target bodies down the tree
 *
 * @param[in]  handleInteraction interaction-handling function
 * @param[in]  bodyIdx_i         target index, TravConfig::nwt per lane
 * @param[in]  pos_i             target x,y,z,4h^2, TravConfig::nwt per lane
 * @param[in]  targetCenter      geometrical target center
 * @param[in]  targetSize        geometrical target bounding box size
 * @param[in]  x,y,z             source bodies as referenced by tree cells
 * @param[in]  tree              octree data view
 * @param[in]  initNodeIdx       traversal will be started with all children of the parent of @p initNodeIdx
 * @param[in]  box               global coordinate bounding box
 * @param[-]   tempQueue         shared mem int pointer to GpuConfig::warpSize ints, uninitialized
 * @param[-]   cellQueue         pointer to global memory, size defined by TravConfig::memPerWarp, uninitialized
 * @return                       Number of P2P interactions tested to the group of target particles.
 *                               The total for the warp is the numbers returned here times the number of valid
 *                               targets in the warp.
 *
 * Constant input pointers are additionally marked __restrict__ to indicate to the compiler that loads
 * can be routed through the read-only/texture cache.
 */
template<bool UsePbc, class InteractionHandler, class Tc, class KeyType>
__device__ uint2 traverseWarp(const InteractionHandler& handleInteraction,
                              const util::array<Vec4<Tc>, TravConfig::nwt>& pos_i,
                              const Vec3<Tc> targetCenter,
                              const Vec3<Tc> targetSize,
                              const Tc* __restrict__ x,
                              const Tc* __restrict__ y,
                              const Tc* __restrict__ z,
                              const OctreeNsView<Tc, KeyType>& tree,
                              int initNodeIdx,
                              const Box<Tc>& box,
                              volatile int* tempQueue,
                              int* cellQueue)
{
    const TreeNodeIndex* __restrict__ childOffsets   = tree.childOffsets;
    const TreeNodeIndex* __restrict__ internalToLeaf = tree.internalToLeaf;
    const LocalIndex* __restrict__ layout            = tree.layout;
    const Vec3<Tc>* __restrict__ centers             = tree.centers;
    const Vec3<Tc>* __restrict__ sizes               = tree.sizes;

    const int laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);

    unsigned p2pCounter = 0, maxStack = 0;

    int bodyQueue; // warp queue for source body indices

    // populate initial cell queue
    if (laneIdx == 0) { cellQueue[0] = initNodeIdx; }

    // these variables are always identical on all warp lanes
    int numSources   = 1; // current stack size
    int newSources   = 0; // stack size for next level
    int oldSources   = 0; // cell indices done
    int sourceOffset = 0; // current level stack pointer, once this reaches numSources, the level is done
    int bdyFillLevel = 0; // fill level of the source body warp queue

    while (numSources > 0) // While there are source cells to traverse
    {
        int sourceIdx   = sourceOffset + laneIdx; // Source cell index of current lane
        int sourceQueue = 0;
        if (laneIdx < GpuConfig::warpSize / 8)
        {
            sourceQueue = cellQueue[ringAddr(oldSources + sourceIdx)]; // Global source cell index in queue
        }
        sourceQueue         = spreadSeg8(sourceQueue);
        sourceIdx           = shflSync(sourceIdx, laneIdx >> 3);
        const bool isSource = sourceIdx < numSources; // Source index is within bounds
        if (!isSource) { sourceQueue = 0; }

        const Vec3<Tc> curSrcCenter = centers[sourceQueue];      // Current source cell center
        const Vec3<Tc> curSrcSize   = sizes[sourceQueue];        // Current source cell center
        const int childBegin        = childOffsets[sourceQueue]; // First child cell
        const bool isNode           = childBegin;
        const bool isClose          = cellOverlap<UsePbc>(curSrcCenter, curSrcSize, targetCenter, targetSize, box);
        const bool isDirect         = isClose && !isNode && isSource;
        const int leafIdx           = (isDirect) ? internalToLeaf[sourceQueue] : 0; // the cstone leaf index

        // Split
        const bool isSplit     = isNode && isClose && isSource;                   // Source cell must be split
        const int numChildLane = exclusiveScanBool(isSplit);                      // Exclusive scan of numChild
        const int numChildWarp = reduceBool(isSplit);                             // Total numChild of current warp
        sourceOffset += imin(GpuConfig::warpSize / 8, numSources - sourceOffset); // advance current level stack pointer
        int childIdx = oldSources + numSources + newSources + numChildLane;       // Child index of current lane
        if (isSplit) { cellQueue[ringAddr(childIdx)] = childBegin; }              // Queue child cells for next level
        newSources += numChildWarp; // Increment source cell count for next loop

        // check for cellQueue overflow
        const unsigned stackUsed = newSources + numSources - sourceOffset; // current cellQueue size
        maxStack                 = max(stackUsed, maxStack);
        if (stackUsed > TravConfig::memPerWarp) { return {0xFFFFFFFF, maxStack}; } // Exit if cellQueue overflows

        // Direct
        const int firstBody     = layout[leafIdx];
        const int numBodies     = (layout[leafIdx + 1] - firstBody) & -int(isDirect); // Number of bodies in cell
        bool directTodo         = numBodies;
        const int numBodiesScan = inclusiveScanInt(numBodies);                      // Inclusive scan of numBodies
        int numBodiesLane       = numBodiesScan - numBodies;                        // Exclusive scan of numBodies
        int numBodiesWarp       = shflSync(numBodiesScan, GpuConfig::warpSize - 1); // Total numBodies of current warp
        int prevBodyIdx         = 0;
        while (numBodiesWarp > 0) // While there are bodies to process from current source cell set
        {
            tempQueue[laneIdx] = 1; // Default scan input is 1, such that consecutive lanes load consecutive bodies
            if (directTodo && (numBodiesLane < GpuConfig::warpSize))
            {
                directTodo               = false;          // Set cell as processed
                tempQueue[numBodiesLane] = -1 - firstBody; // Put first source cell body index into the queue
            }
            const int bodyIdx = inclusiveSegscanInt(tempQueue[laneIdx], prevBodyIdx);
            // broadcast last processed bodyIdx from the last lane to restart the scan in the next iteration
            prevBodyIdx = shflSync(bodyIdx, GpuConfig::warpSize - 1);

            if (numBodiesWarp >= GpuConfig::warpSize) // Process bodies from current set of source cells
            {
                // Load source body coordinates
                const Vec3<Tc> sourceBody = {x[bodyIdx], y[bodyIdx], z[bodyIdx]};
                applyInteraction<UsePbc>(sourceBody, GpuConfig::warpSize, pos_i, box, bodyIdx, handleInteraction);
                numBodiesWarp -= GpuConfig::warpSize;
                numBodiesLane -= GpuConfig::warpSize;
                p2pCounter += GpuConfig::warpSize;
            }
            else // Fewer than warpSize bodies remaining from current source cell set
            {
                // push the remaining bodies into bodyQueue
                int topUp = shflUpSync(bodyIdx, bdyFillLevel);
                bodyQueue = (laneIdx < bdyFillLevel) ? bodyQueue : topUp;

                bdyFillLevel += numBodiesWarp;
                if (bdyFillLevel >= GpuConfig::warpSize) // If this causes bodyQueue to spill
                {
                    // Load source body coordinates
                    const Vec3<Tc> sourceBody = {x[bodyQueue], y[bodyQueue], z[bodyQueue]};
                    applyInteraction<UsePbc>(sourceBody, GpuConfig::warpSize, pos_i, box, bodyQueue, handleInteraction);
                    bdyFillLevel -= GpuConfig::warpSize;
                    // bodyQueue is now empty; put body indices that spilled into the queue
                    bodyQueue = shflDownSync(bodyIdx, numBodiesWarp - bdyFillLevel);
                    p2pCounter += GpuConfig::warpSize;
                }
                numBodiesWarp = 0; // No more bodies to process from current source cells
            }
        }

        //  If the current level is done
        if (sourceOffset >= numSources)
        {
            oldSources += numSources;      // Update finished source size
            numSources   = newSources;     // Update current source size
            sourceOffset = newSources = 0; // Initialize next source size and offset
        }
    }

    if (bdyFillLevel > 0) // If there are leftover direct bodies
    {
        const bool laneHasBody = laneIdx < bdyFillLevel;
        // Load position of source bodies, with padding for invalid lanes
        const Vec3<Tc> sourceBody =
            laneHasBody ? Vec3<Tc>{x[bodyQueue], y[bodyQueue], z[bodyQueue]} : Vec3<Tc>{Tc(0), Tc(0), Tc(0)};
        applyInteraction<UsePbc>(sourceBody, bdyFillLevel, pos_i, box, bodyQueue, handleInteraction);
        p2pCounter += bdyFillLevel;
    }

    return {p2pCounter, maxStack};
}

//! @brief neighbor search traversal statistics: sumP2P, maxP2P, maxStack
struct NcStats
{
    using type = unsigned long long;
    enum IndexNames
    {
        sumP2P,
        maxP2P,
        maxStack,
        numStats
    };
};
static __device__ NcStats::type ncStats[NcStats::numStats];

static __device__ unsigned targetCounterGlob;

static __global__ void resetTraversalCounters()
{
    for (int i = 0; i < NcStats::numStats; ++i)
    {
        ncStats[i] = 0;
    }

    targetCounterGlob = 0;
}

template<class Tc, class ThP, class Index>
__device__ __forceinline__ util::array<Vec4<Tc>, TravConfig::nwt> loadTarget(Index bodyBegin,
                                                                             Index bodyEnd,
                                                                             unsigned lane,
                                                                             const Tc* __restrict__ x,
                                                                             const Tc* __restrict__ y,
                                                                             const Tc* __restrict__ z,
                                                                             const ThP h)
{
    util::array<Vec4<Tc>, TravConfig::nwt> pos_i;
#pragma unroll
    for (unsigned i = 0; i < TravConfig::nwt; i++)
    {
        Index bodyIdx = imin(bodyBegin + i * GpuConfig::warpSize + lane, bodyEnd - 1);
        Tc hi;
        if constexpr (std::is_pointer_v<ThP>)
            hi = h[bodyIdx];
        else
            hi = h;
        pos_i[i] = {x[bodyIdx], y[bodyIdx], z[bodyIdx], Tc(2) * hi};
    }
    return pos_i;
}

//! @brief determine the bounding box around all particles-2h spheres in the warp
template<class Tc>
__device__ __forceinline__ util::tuple<Vec3<Tc>, Vec3<Tc>> warpBbox(const util::array<Vec4<Tc>, TravConfig::nwt>& pos_i)
{
    Tc r0 = pos_i[0][3];
    Vec3<Tc> Xmin{pos_i[0][0] - r0, pos_i[0][1] - r0, pos_i[0][2] - r0};
    Vec3<Tc> Xmax{pos_i[0][0] + r0, pos_i[0][1] + r0, pos_i[0][2] + r0};
#pragma unroll
    for (unsigned i = 1; i < TravConfig::nwt; i++)
    {
        Tc ri = pos_i[i][3];
        Vec3<Tc> iboxMin{pos_i[i][0] - ri, pos_i[i][1] - ri, pos_i[i][2] - ri};
        Vec3<Tc> iboxMax{pos_i[i][0] + ri, pos_i[i][1] + ri, pos_i[i][2] + ri};
        Xmin = min(Xmin, iboxMin);
        Xmax = max(Xmax, iboxMax);
    }

    Xmin = {warpMin(Xmin[0]), warpMin(Xmin[1]), warpMin(Xmin[2])};
    Xmax = {warpMax(Xmax[0]), warpMax(Xmax[1]), warpMax(Xmax[2])};

    Vec3<Tc> targetCenter = (Xmax + Xmin) * Tc(0.5);
    Vec3<Tc> targetSize   = (Xmax - Xmin) * Tc(0.5);

    return {targetCenter, targetSize};
}

/*! @brief Find neighbors of a group of given particles, does not count self reference: min return value is 0
 *
 * @param[in]  bodyBegin   index of first particle in (x,y,z) to look for neighbors
 * @param[in]  bodyEnd     last (excluding) index of particle to look for neighbors
 * @param[in]  x           particle x coordinates
 * @param[in]  y           particle y coordinates
 * @param[in]  z           particle z coordinates
 * @param[in]  h           particle smoothing lengths
 * @param[in]  tree        octree connectivity and cell data
 * @param[in]  box         global coordinate bounding box
 * @param[out] warpNidx    storage for up to ngmax neighbor part. indices for each of the (bodyEnd - bodyBegin) targets
 * @param[in]  ngmax       maximum number of neighbors to store
 * @param[-]   globalPool  global memory for cell traversal stack
 * @return                 actual neighbor count of the particle handled by the executing warp lane, can be > ngmax,
 *                         minimum returned value is 0
 *
 * Note: Number of handled particles (bodyEnd - bodyBegin) should be GpuConfig::warpSize * TravConfig::nwt or smaller
 */
template<class Tc, class ThP, class KeyType>
__device__ util::array<unsigned, TravConfig::nwt> traverseNeighbors(cstone::LocalIndex bodyBegin,
                                                                    cstone::LocalIndex bodyEnd,
                                                                    const Tc* __restrict__ x,
                                                                    const Tc* __restrict__ y,
                                                                    const Tc* __restrict__ z,
                                                                    const ThP h,
                                                                    const OctreeNsView<Tc, KeyType>& tree,
                                                                    const Box<Tc>& box,
                                                                    cstone::LocalIndex* warpNidx,
                                                                    unsigned ngmax,
                                                                    TreeNodeIndex* globalPool)
{
    const unsigned laneIdx                      = threadIdx.x & (GpuConfig::warpSize - 1);
    util::array<unsigned, TravConfig::nwt> nc_i = {};
    auto handleInteraction                      = [&](int warpTarget, cstone::LocalIndex j)
    {
        if (nc_i[warpTarget] < ngmax)
            warpNidx[nc_i[warpTarget] * TravConfig::targetSize + laneIdx + warpTarget * GpuConfig::warpSize] = j;
        ++nc_i[warpTarget];
    };
    traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, handleInteraction, globalPool);

    return nc_i;
}

/*! @brief Find neighbors of a group of given particles, does not count self reference
 *
 * @param[in]  bodyBegin         index of first particle in (x,y,z) to look for neighbors
 * @param[in]  bodyEnd           last (excluding) index of particle to look for neighbors
 * @param[in]  x                 particle x coordinates
 * @param[in]  y                 particle y coordinates
 * @param[in]  z                 particle z coordinates
 * @param[in]  h                 particle smoothing lengths
 * @param[in]  tree              octree connectivity and cell data
 * @param[in]  box               global coordinate bounding box
 * @param[in]  handleInteraction interaction-handling function
 * @param[-]   globalPool        global memory for cell traversal stack
 *
 * Note: Number of handled particles (bodyEnd - bodyBegin) should be GpuConfig::warpSize * TravConfig::nwt or smaller
 */
template<class Tc, class ThP, class KeyType, class InteractionHandler>
__device__ void traverseNeighbors(cstone::LocalIndex bodyBegin,
                                  cstone::LocalIndex bodyEnd,
                                  const Tc* __restrict__ x,
                                  const Tc* __restrict__ y,
                                  const Tc* __restrict__ z,
                                  const ThP h,
                                  const OctreeNsView<Tc, KeyType>& tree,
                                  const Box<Tc>& box,
                                  const InteractionHandler& handleInteraction,
                                  TreeNodeIndex* globalPool)
{
    const unsigned laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
    const unsigned warpIdx = threadIdx.x >> GpuConfig::warpSizeLog2;

    constexpr unsigned numWarpsPerBlock = TravConfig::numThreads / GpuConfig::warpSize;

    __shared__ int sharedPool[TravConfig::numThreads];

    // warp-common shared mem, 1 int per thread
    int* tempQueue = sharedPool + GpuConfig::warpSize * warpIdx;
    // warp-common global mem storage
    int* cellQueue = globalPool + TravConfig::memPerWarp * ((blockIdx.x * numWarpsPerBlock) + warpIdx);

    util::array<Vec4<Tc>, TravConfig::nwt> pos_i = loadTarget(bodyBegin, bodyEnd, laneIdx, x, y, z, h);
    auto [targetCenter, targetSize]              = warpBbox(pos_i);
    targetSize *= Tc(tree.searchExtFactor);

#pragma unroll
    for (unsigned k = 0; k < TravConfig::nwt; ++k)
    {
        auto r      = pos_i[k][3];
        pos_i[k][3] = r * r;
    }

    auto pbc    = BoundaryType::periodic;
    bool anyPbc = box.boundaryX() == pbc || box.boundaryY() == pbc || box.boundaryZ() == pbc;
    bool usePbc = anyPbc && !insideBox(targetCenter, targetSize, box);

    // start traversal with node 1 (first child of the root), implies siblings as well
    // if traversal should be started at node x, then initNode should be set to the first child of x
    int initNode = 1;

    uint2 warpStats;
    if (usePbc)
    {
        warpStats = traverseWarp<true>(handleInteraction, pos_i, targetCenter, targetSize, x, y, z, tree, initNode, box,
                                       tempQueue, cellQueue);
    }
    else
    {
        warpStats = traverseWarp<false>(handleInteraction, pos_i, targetCenter, targetSize, x, y, z, tree, initNode,
                                        box, tempQueue, cellQueue);
    }
    unsigned numP2P   = warpStats.x;
    unsigned maxStack = warpStats.y;
    assert(numP2P != 0xFFFFFFFF);

    if (laneIdx == 0)
    {
        unsigned targetGroupSize = bodyEnd - bodyBegin;
        atomicAdd(&ncStats[NcStats::sumP2P], NcStats::type(numP2P) * targetGroupSize);
        atomicMax(&ncStats[NcStats::maxP2P], NcStats::type(numP2P));
        atomicMax(&ncStats[NcStats::maxStack], NcStats::type(maxStack));
    }
}

//! @brief combine temp space for tree traversal and neighbor search into a single allocation
template<class DeviceVector>
std::tuple<TreeNodeIndex*, LocalIndex*> allocateNcStacks(DeviceVector& stack, unsigned ngmax)
{
    unsigned poolSize = TravConfig::poolSize();
    unsigned nidxSize = ngmax * TravConfig::numBlocks() * TravConfig::numThreads;

    static_assert(sizeof(LocalIndex) == sizeof(typename DeviceVector::value_type));
    reallocateDestructive(stack, poolSize + nidxSize, 1.01);

    auto* traversalPool = reinterpret_cast<TreeNodeIndex*>(rawPtr(stack));
    auto* nidxPool      = reinterpret_cast<LocalIndex*>(rawPtr(stack)) + poolSize;

    return {traversalPool, nidxPool};
}

} // namespace cstone
