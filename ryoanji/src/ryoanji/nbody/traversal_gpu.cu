/*
 * MIT License
 *
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Barnes-Hut breadth-first warp-aware tree traversal inspired by the original Bonsai implementation
 *
 * @author Rio Yokota <rioyokota@gsic.titech.ac.jp>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/cuda/gpu_config.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/primitives/warpscan.cuh"
#include "cartesian_qpole.hpp"
#include "kernel.hpp"
#include "traversal_gpu.h"

namespace ryoanji
{

using cstone::ballotSync;
using cstone::GpuConfig;
using cstone::imin;
using cstone::inclusiveScanInt;
using cstone::inclusiveSegscanInt;
using cstone::shflDownSync;
using cstone::shflSync;
using cstone::shflUpSync;
using cstone::shflXorSync;
using cstone::spreadSeg8;
using cstone::streamCompact;
using cstone::syncWarp;
using cstone::warpMax;
using cstone::warpMin;

struct TravConfig
{
    //! @brief size of global workspace memory per warp, must be a power of 2
    static constexpr int memPerWarp = 512 * GpuConfig::warpSize;
    static_assert((memPerWarp & (memPerWarp - 1)) == 0);

    //! @brief number of threads per block for the traversal kernel
    static constexpr int numThreads = 256;

    static constexpr int numWarpsPerSm = 20;
    //! @brief maximum number of simultaneously active blocks
    inline static int maxNumActiveBlocks =
        GpuConfig::smCount * (TravConfig::numWarpsPerSm / (TravConfig::numThreads / GpuConfig::warpSize));

    //! @brief number of particles per target, i.e. per warp
    static constexpr int targetSize = 64;

    //! @brief number of warps per target, used all over the place, hence the short name
    static constexpr int nwt = targetSize / GpuConfig::warpSize;
};

__device__ __forceinline__ int ringAddr(const int i) { return i & (TravConfig::memPerWarp - 1); }

/*! @brief evaluate multipole acceptance criterion
 *
 * @tparam T
 * @param sourceCenter  center-mass coordinates of the source cell
 * @param MAC           the square of the MAC acceptance radius around @p sourceCenter
 * @param targetCenter  geometric center of the target particle group bounding box
 * @param targetSize    the half length in each dimension of the target bounding box
 * @return              true if the target is too close for using the multipole
 */
template<class T>
__host__ __device__ __forceinline__ bool applyMAC(Vec3<T> sourceCenter, T MAC, Vec3<T> targetCenter, Vec3<T> targetSize)
{
    Vec3<T> dX = abs(targetCenter - sourceCenter) - targetSize;
    dX += abs(dX);
    dX *= T(0.5);
    return norm2(dX) < MAC;
}

/*! @brief apply M2P kernel with GpuConfig::warpSize different multipoles to the target bodies
 *
 * @tparam       T            float or double
 * @tparam       MType        Multipole type, e.g. Cartesian or Spherical of varying expansion order
 * @param[inout] acc_i        target particle acceleration to add to
 * @param[in]    pos_i        target particle (x,y,z,h) of each lane
 * @param[in]    cellIdx      the index of each lane of the multipole to apply, in [0:numSourceCells]
 * @param[in]    srcCenter    pointer to source cell centers in global memory, length numSourceCells
 * @param[in]    Multipoles   pointer to the multipole array in global memory, length numSourceCells
 * @param[-]     warpSpace    shared memory for temporary multipole storage, uninitialized
 *
 * Number of computed M2P interactions per call is GpuConfig::warpSize^2 * TravConfig::nwt
 */
template<class Ta, class Tc, class Tf, class MType>
__device__ void approxAcc(Vec4<Ta> acc_i[TravConfig::nwt], const Vec4<Tc> pos_i[TravConfig::nwt], const int cellIdx,
                          const Vec4<Tf>* __restrict__ srcCenter, const MType* __restrict__ Multipoles,
                          volatile int* warpSpace)
{
    constexpr int termSize = MType{}.size();
    static_assert(termSize <= GpuConfig::warpSize, "multipole size too large for shared-mem warpSpace");

    using MValueType = typename MType::value_type;

    auto* sm_Multipole      = reinterpret_cast<volatile MValueType*>(warpSpace);
    auto* __restrict__ gm_M = reinterpret_cast<const MValueType*>(Multipoles);

    const int laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);

    for (int j = 0; j < GpuConfig::warpSize; j++)
    {
        int currentCell = shflSync(cellIdx, j);
        if (currentCell < 0) { continue; }

        Vec3<Tf> pos_j = makeVec3(srcCenter[currentCell]);

        if (laneIdx < termSize) { sm_Multipole[laneIdx] = gm_M[currentCell * termSize + laneIdx]; }
        syncWarp();

#pragma unroll
        for (int k = 0; k < TravConfig::nwt; k++)
        {
            acc_i[k] = M2P(acc_i[k], makeVec3(pos_i[k]), pos_j, *(MType*)sm_Multipole);
        }
    }
}

/*! @brief compute body-body interactions
 *
 * @tparam       T            float or double
 * @param[in]    sourceBody   source body x,y,z,m
 * @param[in]    hSource      source body smoothing length
 * @param[inout] acc_i        target acceleration to add to
 * @param[in]    pos_i        target body x,y,z,h
 *
 * Number of computed P2P interactions per call is GpuConfig::warpSize^2 * TravConfig::nwt
 */
template<class Tc, class Th, class Ta>
__device__ void directAcc(Vec4<Tc> sourceBody, Th hSource, Vec4<Ta> acc_i[TravConfig::nwt],
                          const Vec4<Tc> pos_i[TravConfig::nwt])
{
    for (int j = 0; j < GpuConfig::warpSize; j++)
    {
        Vec3<Tc> pos_j{shflSync(sourceBody[0], j), shflSync(sourceBody[1], j), shflSync(sourceBody[2], j)};
        Tc       m_j = shflSync(sourceBody[3], j);
        Th       h_j = shflSync(hSource, j);

#pragma unroll
        for (int k = 0; k < TravConfig::nwt; k++)
        {
            acc_i[k] = P2P(acc_i[k], makeVec3(pos_i[k]), pos_j, m_j, h_j, Th(pos_i[k][3]));
        }
    }
}

/*! @brief traverse one warp with up to 64 target bodies down the tree
 *
 * @param[inout] acc_i          acceleration of target to add to, TravConfig::nwt Vec4 per lane
 * @param[in]    pos_i          target positions, and smoothing lengths, TravConfig::nwt per lane
 * @param[in]    targetCenter   geometrical target center
 * @param[in]    targetSize     geometrical target bounding box size
 * @param[in]    x,y,z,m,h      source bodies as referenced by tree cells
 * @param[in]    childOffsets   location (index in [0:numTreeNodes]) of first child of each cell, 0 indicates a leaf
 * @param[in]    internalToLeaf for each cell in [0:numTreeNodes], stores the leaf cell (cstone) index in [0:numLeaves]
 *                              if the cell is not a leaf, the value is negative
 * @param[in]    layout         for each leaf cell in [0:numLeaves], stores the index of the first body in the cell
 * @param[in]    sourceCenter   x,y,z center and square MAC radius of each cell in [0:numTreeNodes]
 * @param[in]    Multipoles     the multipole expansions in the same order as srcCells
 * @param[in]    initNodeIdx    source cell indices indices of the top 8 octants
 * @param[-]     tempQueue      shared mem int pointer to GpuConfig::warpSize ints, uninitialized
 * @param[-]     cellQueue      pointer to TravConfig::memPerWarp ints of warps-specific space global memory
 * @return                      Number of M2P and P2P interactions applied to the group of target particles.
 *                              The totals for the warp are the numbers returned here times the number of valid
 *                              targets in the warp.
 *
 * Constant input pointers are additionally marked __restrict__ to indicate to the compiler that loads
 * can be routed through the read-only/texture cache.
 */
template<class Tc, class Th, class Tm, class Ta, class Tf, class MType>
__device__ util::tuple<unsigned, unsigned, unsigned>
           traverseWarp(Vec4<Ta>* acc_i, const Vec4<Tc> pos_i[TravConfig::nwt], const Vec3<Tf> targetCenter,
                        const Vec3<Tf> targetSize, const Tc* __restrict__ x, const Tc* __restrict__ y, const Tc* __restrict__ z,
                        const Tm* __restrict__ m, const Th* __restrict__ h, const TreeNodeIndex* __restrict__ childOffsets,
                        const TreeNodeIndex* __restrict__ internalToLeaf, const LocalIndex* __restrict__ layout,
                        const Vec4<Tf>* __restrict__ sourceCenter, const MType* __restrict__ Multipoles, int initNodeIdx,
                        volatile int* tempQueue, int* cellQueue)
{
    const int laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);

    unsigned p2pCounter = 0, m2pCounter = 0, maxStack = 0;

    int approxQueue; // warp queue for multipole approximation cell indices
    int bodyQueue;   // warp queue for source body indices

    // populate initial cell queue
    if (laneIdx == 0) { cellQueue[0] = initNodeIdx; }

    // these variables are always identical on all warp lanes
    int numSources   = 1; // current stack size
    int newSources   = 0; // stack size for next level
    int oldSources   = 0; // cell indices done
    int sourceOffset = 0; // current level stack pointer, once this reaches numSources, the level is done
    int apxFillLevel = 0; // fill level of the multipole approximation warp queue
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

        const Vec4<Tf> MAC = sourceCenter[sourceQueue];        // load source cell center + MAC
        const Vec3<Tf> curSrcCenter{MAC[0], MAC[1], MAC[2]};   // Current source cell center
        const int      childBegin = childOffsets[sourceQueue]; // First child cell
        const bool     isNode     = childBegin;
        const bool     isClose    = applyMAC(curSrcCenter, MAC[3], targetCenter, targetSize); // Is too close for MAC
        const bool     isDirect   = isClose && !isNode && isSource;
        const int      leafIdx    = (isDirect) ? internalToLeaf[sourceQueue] : 0; // the cstone leaf index

        // Split
        const bool isSplit      = isNode && isClose && isSource;                  // Source cell must be split
        const int  numChildLane = cstone::exclusiveScanBool(isSplit);             // Exclusive scan of numChild
        const int  numChildWarp = cstone::reduceBool(isSplit);                    // Total numChild of current warp
        sourceOffset += imin(GpuConfig::warpSize / 8, numSources - sourceOffset); // advance current level stack pointer
        int childIdx = oldSources + numSources + newSources + numChildLane;       // Child index of current lane
        if (isSplit) { cellQueue[ringAddr(childIdx)] = childBegin; }              // Queue child cells for next level
        newSources += numChildWarp; //  Increment source cell count for next loop

        // check for cellQueue overflow
        const unsigned stackUsed = newSources + numSources - sourceOffset;
        maxStack                 = max(stackUsed, maxStack);
        if (stackUsed > TravConfig::memPerWarp) { return {0xFFFFFFFF, 0xFFFFFFFF, maxStack}; }

        // Multipole approximation
        const bool isRemote    = layout[leafIdx + 1] == layout[leafIdx] && Multipoles[sourceQueue][Cqi::mass] > 0;
        const bool isApprox    = (!isClose || (!isNode && isRemote)) && isSource; // Source cell can be used for M2P
        int        numKeepWarp = streamCompact(&sourceQueue, isApprox, tempQueue);
        // push valid approx source cell indices into approxQueue
        const int apxTopUp = shflUpSync(sourceQueue, apxFillLevel);
        approxQueue        = (laneIdx < apxFillLevel) ? approxQueue : apxTopUp;
        apxFillLevel += numKeepWarp;

        if (apxFillLevel >= GpuConfig::warpSize) // If queue is larger than warp size,
        {
            // Call M2P kernel
            approxAcc(acc_i, pos_i, approxQueue, sourceCenter, Multipoles, tempQueue);
            apxFillLevel -= GpuConfig::warpSize;
            // pull down remaining source cell indices into now empty approxQueue
            approxQueue = shflDownSync(sourceQueue, numKeepWarp - apxFillLevel);
            m2pCounter += warpSize;
        }

        // Direct
        const int firstBody     = layout[leafIdx];
        const int numBodies     = (layout[leafIdx + 1] - firstBody) & -int(isDirect); // Number of bodies in cell
        bool      directTodo    = numBodies;
        const int numBodiesScan = inclusiveScanInt(numBodies);                      // Inclusive scan of numBodies
        int       numBodiesLane = numBodiesScan - numBodies;                        // Exclusive scan of numBodies
        int       numBodiesWarp = shflSync(numBodiesScan, GpuConfig::warpSize - 1); // Total numBodies of current warp
        int       prevBodyIdx   = 0;
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
                const Vec4<Tc> sourceBody = {x[bodyIdx], y[bodyIdx], z[bodyIdx], Tc(m[bodyIdx])};
                const Tm       hSource    = h[bodyIdx];
                directAcc(sourceBody, hSource, acc_i, pos_i);
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
                    const Vec4<Tc> sourceBody = {x[bodyQueue], y[bodyQueue], z[bodyQueue], Tc(m[bodyQueue])};
                    const Tm       hSource    = h[bodyQueue];
                    directAcc(sourceBody, hSource, acc_i, pos_i);
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

    if (apxFillLevel > 0) // If there are leftover approx cells
    {
        // Call M2P kernel
        approxAcc(acc_i, pos_i, laneIdx < apxFillLevel ? approxQueue : -1, sourceCenter, Multipoles, tempQueue);

        m2pCounter += apxFillLevel;
    }

    if (bdyFillLevel > 0) // If there are leftover direct bodies
    {
        const int bodyIdx = laneIdx < bdyFillLevel ? bodyQueue : -1;
        // Load position of source bodies, with padding for invalid lanes
        const Vec4<Tc> sourceBody = bodyIdx >= 0 ? Vec4<Tc>{x[bodyIdx], y[bodyIdx], z[bodyIdx], Tc(m[bodyIdx])}
                                                 : Vec4<Tc>{Tc(0), Tc(0), Tc(0), Tc(0)};
        const Tm       hSource    = bodyIdx >= 0 ? h[bodyIdx] : Tm(0);
        directAcc(sourceBody, hSource, acc_i, pos_i);
        p2pCounter += bdyFillLevel;
    }

    return {m2pCounter, p2pCounter, maxStack};
}

//! @brief Barnes-Hut traversal statistics: sumP2P, maxP2P, maxStack
struct BhStats
{
    using type = unsigned long long;
    enum IndexNames
    {
        sumP2P,
        maxP2P,
        sumM2P,
        maxM2P,
        maxStack,
        numStats
    };
};
static __device__ BhStats::type bhStats[BhStats::numStats];

__device__ float        totalPotentialGlob = 0;
__device__ unsigned int targetCounterGlob  = 0;

__global__ void resetTraversalCounters()
{
    for (int i = 0; i < BhStats::numStats; ++i)
    {
        bhStats[i] = 0;
    }

    totalPotentialGlob = 0;
    targetCounterGlob  = 0;
}

/*! @brief Compute approximate body accelerations with Barnes-Hut
 *
 * @param[in]    grp            groups of target particles to compute accelerations for
 * @param[in]    initNodeIdx    traversal will be started with all children of the parent of @p initNodeIdx
 * @param[in]    xt,yt,zt,mt,ht target bodies, as referenced and grouped by @p grp
 * @param[in]    xs,ys,zs,ms,hs source bodies, in SFC order as used for tree-build and as referenced by layout
 * @param[in]    childOffsets   location (index in [0:numTreeNodes]) of first child of each cell, 0 indicates a leaf
 * @param[in]    internalToLeaf for each cell in [0:numTreeNodes], stores the leaf cell (cstone) index in [0:numLeaves]
 *                              if the cell is not a leaf, the value is negative
 * @param[in]    layout         for each leaf cell in [0:numLeaves], stores the index of the first source body in cell
 * @param[in]    sourceCenter   x,y,z center and square MAC radius of each cell in [0:numTreeNodes]
 * @param[in]    Multipoles     cell multipoles, on device
 * @param[in]    G              gravitational constant
 * @param[in]    numShells      number of periodic replicas in each dimension to include
 * @param[in]    boxL           length of coordinate bounding box in each dimension
 * @param[inout] p              output body potential to add to if not nullptr
 * @param[inout] ax, ay, az     output body acceleration to add to
 * @param[-]     gmPool         temporary storage for the cell traversal stack, uninitialized
 *                              each active warp needs space for TravConfig::memPerWarp int32,
 *                              so the total size is TravConfig::memPerWarp * numWarpsPerBlock * numBlocks
 */
template<class Tc, class Th, class Tm, class Ta, class Tf, class MType>
__global__ __launch_bounds__(TravConfig::numThreads) void traverseKernel(
    cstone::GroupView grp, const int initNodeIdx, const Tc* __restrict__ xt, const Tc* __restrict__ yt,
    const Tc* __restrict__ zt, const Tm* __restrict__ mt, const Th* __restrict__ ht, const Tc* __restrict__ xs,
    const Tc* __restrict__ ys, const Tc* __restrict__ zs, const Tm* __restrict__ ms, const Th* __restrict__ hs,
    const TreeNodeIndex* __restrict__ childOffsets, const TreeNodeIndex* __restrict__ internalToLeaf,
    const LocalIndex* __restrict__ layout, const Vec4<Tf>* __restrict__ sourceCenter,
    const MType* __restrict__ Multipoles, Tc G, int numShells, Vec3<Tc> boxL, Ta* p, Ta* ax, Ta* ay, Ta* az,
    int* gmPool)
{
    const int laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
    const int warpIdx = threadIdx.x >> GpuConfig::warpSizeLog2;

    constexpr int termSize         = MType{}.size();
    constexpr int numWarpsPerBlock = TravConfig::numThreads / GpuConfig::warpSize;

    using MValueType         = typename MType::value_type;
    constexpr int mSizeRatio = sizeof(MValueType) / sizeof(int);

    static_assert(termSize <= GpuConfig::warpSize, "review approxAcc function before disabling this check");
    constexpr int smSize =
        (TravConfig::numThreads > termSize * numWarpsPerBlock) ? TravConfig::numThreads : termSize * numWarpsPerBlock;
    __shared__ int sharedPool[smSize * mSizeRatio];

    // warp-common shared mem, 1 int per thread
    int* tempQueue = sharedPool + GpuConfig::warpSize * warpIdx * mSizeRatio;
    // warp-common global mem storage
    int* cellQueue = gmPool + TravConfig::memPerWarp * ((blockIdx.x * numWarpsPerBlock) + warpIdx);

    int targetIdx = 0;

    while (true)
    {
        // first thread in warp grabs next target
        if (laneIdx == 0)
        {
            // this effectively randomizes which warp gets which targets, which better balances out
            // the load imbalance between different targets compared to static assignment
            targetIdx = atomicAdd(&targetCounterGlob, 1);
        }
        targetIdx = shflSync(targetIdx, 0);

        if (targetIdx >= grp.numGroups) return;

        const int bodyBegin = grp.groupStart[targetIdx];
        const int bodyEnd   = grp.groupEnd[targetIdx];

        // load target coordinates
        Vec4<Tc> pos_i[TravConfig::nwt];
        for (int i = 0; i < TravConfig::nwt; i++)
        {
            int bodyIdx = imin(bodyBegin + i * GpuConfig::warpSize + laneIdx, bodyEnd - 1);
            pos_i[i]    = {xt[bodyIdx], yt[bodyIdx], zt[bodyIdx], ht[bodyIdx]};
        }

        Vec3<Tc> Xmin = makeVec3(pos_i[0]);
        Vec3<Tc> Xmax = makeVec3(pos_i[0]);
        for (int i = 1; i < TravConfig::nwt; i++)
        {
            Xmin = min(Xmin, makeVec3(pos_i[i]));
            Xmax = max(Xmax, makeVec3(pos_i[i]));
        }

        Xmin = {warpMin(Xmin[0]), warpMin(Xmin[1]), warpMin(Xmin[2])};
        Xmax = {warpMax(Xmax[0]), warpMax(Xmax[1]), warpMax(Xmax[2])};

        Vec3<Tf>       targetCenter = (Xmax + Xmin) * Tf(0.5);
        const Vec3<Tf> targetSize   = (Xmax - Xmin) * Tf(0.5);

        Vec4<Tf> acc_i[TravConfig::nwt];
        for (int i = 0; i < TravConfig::nwt; i++)
        {
            acc_i[i] = Vec4<Tc>{Tc(0), Tc(0), Tc(0), Tc(0)};
        }

        unsigned numM2P = 0, numP2P = 0, maxStack = 0;
        for (int iz = -numShells; iz <= numShells; ++iz)
        {
            for (int iy = -numShells; iy <= numShells; ++iy)
            {
                for (int ix = -numShells; ix <= numShells; ++ix)
                {
                    {
                        Vec3<Tf> pbcShift{ix * boxL[0], iy * boxL[1], iz * boxL[2]};
                        targetCenter += pbcShift;
                        for (int i = 0; i < TravConfig::nwt; i++)
                        {
                            pos_i[i][0] += pbcShift[0];
                            pos_i[i][1] += pbcShift[1];
                            pos_i[i][2] += pbcShift[2];
                        }
                    }

                    auto [numM2P_, numP2P_, maxStack_] = traverseWarp(
                        acc_i, pos_i, targetCenter, targetSize, xs, ys, zs, ms, hs, childOffsets, internalToLeaf,
                        layout, sourceCenter, Multipoles, initNodeIdx, tempQueue, cellQueue);

                    {
                        Vec3<Tf> pbcShift{ix * boxL[0], iy * boxL[1], iz * boxL[2]};
                        targetCenter -= pbcShift;
                        for (int i = 0; i < TravConfig::nwt; i++)
                        {
                            pos_i[i][0] -= pbcShift[0];
                            pos_i[i][1] -= pbcShift[1];
                            pos_i[i][2] -= pbcShift[2];
                        }
                    }

                    assert(numM2P_ != 0xFFFFFFFF && numP2P_ != 0xFFFFFFFF);
                    numM2P += numM2P_;
                    numP2P += numP2P_;
                    maxStack = max(maxStack, maxStack_);
                }
            }
        }

        const int bodyIdxLane = bodyBegin + laneIdx;

        float warpPotential = 0;
        for (int i = 0; i < TravConfig::nwt; i++)
        {
            const int bodyIdx = bodyIdxLane + i * GpuConfig::warpSize;
            if (bodyIdx < bodyEnd) { warpPotential += mt[bodyIdx] * acc_i[i][0]; }
        }

#pragma unroll
        for (int i = 0; i < GpuConfig::warpSizeLog2; i++)
        {
            warpPotential += shflXorSync(warpPotential, 1 << i);
        }

        if (laneIdx == 0)
        {
            int targetGroupSize = bodyEnd - bodyBegin;
            atomicMax(&bhStats[BhStats::maxP2P], numP2P);
            atomicAdd(&bhStats[BhStats::sumP2P], numP2P * targetGroupSize);
            atomicMax(&bhStats[BhStats::maxM2P], numM2P);
            atomicAdd(&bhStats[BhStats::sumM2P], numM2P * targetGroupSize);
            atomicMax(&bhStats[BhStats::maxStack], maxStack);
            atomicAdd(&totalPotentialGlob, warpPotential);
        }

        for (int i = 0; i < TravConfig::nwt; i++)
        {
            const int bodyIdx = bodyIdxLane + i * GpuConfig::warpSize;
            if (bodyIdx < bodyEnd)
            {
                if (p) { p[bodyIdx] += G * mt[bodyIdx] * acc_i[i][0]; }
                ax[bodyIdx] += G * acc_i[i][1];
                ay[bodyIdx] += G * acc_i[i][2];
                az[bodyIdx] += G * acc_i[i][3];
            }
        }
    }
}

template<class Tc, class Th, class Tm, class Ta, class Tf, class MType>
double traverse(cstone::GroupView grp, const int initNodeIdx, const Tc* __restrict__ xt, const Tc* __restrict__ yt,
                const Tc* __restrict__ zt, const Tm* __restrict__ mt, const Th* __restrict__ ht,
                const Tc* __restrict__ xs, const Tc* __restrict__ ys, const Tc* __restrict__ zs,
                const Tm* __restrict__ ms, const Th* __restrict__ hs, const TreeNodeIndex* __restrict__ childOffsets,
                const TreeNodeIndex* __restrict__ internalToLeaf, const LocalIndex* __restrict__ layout,
                const Vec4<Tf>* __restrict__ sourceCenter, const MType* __restrict__ multipoles, Tc G, int numShells,
                Vec3<Tc> boxL, Ta* p, Ta* ax, Ta* ay, Ta* az, int* gmPool)
{

    int numWarpsPerBlock = TravConfig::numThreads / GpuConfig::warpSize;
    int numBlocks        = cstone::iceil(grp.numGroups, numWarpsPerBlock);
    numBlocks            = std::min(numBlocks, TravConfig::maxNumActiveBlocks);

    resetTraversalCounters<<<1, 1>>>();
    traverseKernel<<<numBlocks, TravConfig::numThreads>>>(grp, initNodeIdx, xt, yt, zt, mt, ht, xs, ys, zs, ms, hs,
                                                          childOffsets, internalToLeaf, layout, sourceCenter,
                                                          multipoles, G, numShells, boxL, p, ax, ay, az, gmPool);
    float totalPotential;
    checkGpuErrors(cudaMemcpyFromSymbol(&totalPotential, GPU_SYMBOL(totalPotentialGlob), sizeof(float)));
    return 0.5 * Tc(G) * totalPotential;
}

#define TRAVERSE(Tc, Th, Tm, Ta, Tf, MType)                                                                            \
    template double traverse(cstone::GroupView grp, const int initNodeIdx, const Tc* __restrict__ xt,                  \
                             const Tc* __restrict__ yt, const Tc* __restrict__ zt, const Tm* __restrict__ mt,          \
                             const Th* __restrict__ ht, const Tc* __restrict__ xs, const Tc* __restrict__ ys,          \
                             const Tc* __restrict__ zs, const Tm* __restrict__ ms, const Th* __restrict__ hs,          \
                             const TreeNodeIndex* __restrict__ childOffsets,                                           \
                             const TreeNodeIndex* __restrict__ internalToLeaf, const LocalIndex* __restrict__ layout,  \
                             const Vec4<Tf>* __restrict__ sourceCenter, const MType* __restrict__ multipoles, Tc G,    \
                             int numShells, Vec3<Tc> boxL, Ta* p, Ta* ax, Ta* ay, Ta* az, int* gmPool)

#define TRAVERSE_MPOLE(MType)                                                                                          \
    TRAVERSE(double, double, double, double, double, MType<double>);                                                   \
    TRAVERSE(double, float, float, float, double, MType<float>);                                                       \
    TRAVERSE(float, float, float, float, float, MType<float>);

TRAVERSE_MPOLE(CartesianQuadrupole)
TRAVERSE_MPOLE(CartesianMDQpole)

int bhMaxTargetSize() { return TravConfig::targetSize; }

LocalIndex stackSize(LocalIndex numGroups)
{
    int numWarpsPerBlock = TravConfig::numThreads / cstone::GpuConfig::warpSize;
    int numBlocks        = cstone::iceil(numGroups, numWarpsPerBlock);
    numBlocks            = std::min(numBlocks, TravConfig::maxNumActiveBlocks);
    LocalIndex poolSize  = TravConfig::memPerWarp * numWarpsPerBlock * numBlocks;
    return poolSize;
}

util::array<uint64_t, 5> readBhStats()
{
    BhStats::type stats[BhStats::numStats];
    checkGpuErrors(cudaMemcpyFromSymbol(stats, GPU_SYMBOL(bhStats), BhStats::numStats * sizeof(BhStats::type)));

    auto sumP2P   = stats[BhStats::sumP2P];
    auto maxP2P   = stats[BhStats::maxP2P];
    auto sumM2P   = stats[BhStats::sumM2P];
    auto maxM2P   = stats[BhStats::maxM2P];
    auto maxStack = stats[BhStats::maxStack];

    return {sumP2P, maxP2P, sumM2P, maxM2P, maxStack};
}

} // namespace ryoanji
