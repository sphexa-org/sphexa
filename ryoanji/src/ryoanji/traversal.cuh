/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
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

#pragma once

#include <algorithm>
#include "kernel.hpp"
#include "warpscan.cuh"

namespace ryoanji
{

struct TravConfig
{
    //! @brief size of global workspace memory per warp
    static constexpr int memPerWarp = 2048 * GpuConfig::warpSize;
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

__host__ __device__ __forceinline__ bool applyMAC(fvec3 sourceCenter, float MAC, CellData sourceData,
                                                  fvec3 targetCenter, fvec3 targetSize)
{
    fvec3 dX = abs(targetCenter - sourceCenter) - targetSize;
    dX += abs(dX);
    dX *= 0.5f;
    const float R2 = norm2(dX);
    return R2 < fabsf(MAC) || sourceData.nbody() < 3;
}

//! @brief apply M2P kernel for WarpSize different multipoles to the warp-owned target bodies
__device__ void approxAcc(fvec4 acc_i[TravConfig::nwt], const fvec3 pos_i[TravConfig::nwt], const int cellIdx,
                          const fvec4* __restrict__ srcCenter, const fvec4* __restrict__ Multipoles, const float EPS2,
                          volatile int* warpSpace)
{
    static_assert(NTERM <= GpuConfig::warpSize, "needs adaptation to work beyond octopoles");

    auto sm_Multipole              = reinterpret_cast<volatile float*>(warpSpace);
    const float* __restrict__ gm_M = reinterpret_cast<const float*>(Multipoles);

    const int laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);

    for (int j = 0; j < GpuConfig::warpSize; j++)
    {
        int currentCell = shflSync(cellIdx, j);
        if (currentCell < 0) { continue; }

        fvec3 pos_j = make_fvec3(srcCenter[currentCell]);

        if (laneIdx < NTERM) { sm_Multipole[laneIdx] = gm_M[currentCell * NTERM + laneIdx]; }
        syncWarp();

        for (int k = 0; k < TravConfig::nwt; k++)
            acc_i[k] = M2P(acc_i[k], pos_i[k], pos_j, *(fvecP*)sm_Multipole, EPS2);
    }
}

//! @brief compute body-body interactions
__device__ void directAcc(fvec4 sourceBody, fvec4 acc_i[TravConfig::nwt], const fvec3 pos_i[TravConfig::nwt],
                          const float EPS2)
{
    for (int j = 0; j < GpuConfig::warpSize; j++)
    {
        fvec3 pos_j{shflSync(sourceBody[0], j), shflSync(sourceBody[1], j), shflSync(sourceBody[2], j)};
        float q_j = shflSync(sourceBody[3], j);

        #pragma unroll
        for (int k = 0; k < TravConfig::nwt; k++)
        {
            acc_i[k] = P2P(acc_i[k], pos_i[k], pos_j, q_j, EPS2);
        }
    }
}

/*! @brief traverse one warp with up to 64 target bodies down the tree
 *
 * @param[inout] acc_i         acceleration to add to, TravConfig::nwt fvec4 per lane
 * @param[in]    pos_i         target positions, TravConfig::nwt per lane
 * @param[in]    targetCenter  geometrical target center
 * @param[in]    targetSize    geometrical target bounding box size
 * @param[in]    bodyPos       source bodies as referenced by tree cells
 * @param[in]    sourceCells   source cell data
 * @param[in]    sourceCenter  source center data, x,y,z location and square of MAC radius, same order as sourceCells
 * @param[in]    Multipoles    the multipole expansions in the same order as srcCells
 * @param[in]    EPS2          plummer softening
 * @param[in]    rootRange     source cell indices indices of the top 8 octants
 * @param[-]     tempQueue     shared mem int pointer to 32 ints, uninitialized
 * @param[-]     cellQueue     shared mem int pointer to global memory, 4096 ints per thread, uninitialized
 * @return
 *
 * Constant input pointers are additionally marked __restrict__ to indicate to the compiler that loads
 * can be routed through the read-only/texture cache.
 */
__device__ uint2 traverseWarp(fvec4* acc_i, const fvec3 pos_i[TravConfig::nwt], const fvec3 targetCenter,
                              const fvec3 targetSize, const fvec4* __restrict__ bodyPos,
                              const CellData* __restrict__ sourceCells, const fvec4* __restrict__ sourceCenter,
                              const fvec4* __restrict__ Multipoles, const float EPS2, int2 rootRange,
                              volatile int* tempQueue, int* cellQueue)
{
    const int laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);

    unsigned p2pCounter = 0;
    unsigned m2pCounter = 0;

    int approxQueue; // warp queue for multipole approximation cell indices
    int bodyQueue;   // warp queue for source body indices

    // populate initial cell queue
    for (int root = rootRange.x; root < rootRange.y; root += GpuConfig::warpSize)
    {
        if (root + laneIdx < rootRange.y)
        {
            cellQueue[ringAddr(root - rootRange.x + laneIdx)] = root + laneIdx;
        }
    }

    // these variables are always identical on all warp lanes
    int numSources   = rootRange.y - rootRange.x; // current stack size
    int newSources   = 0; // stack size for next level
    int oldSources   = 0; // cell indices done
    int sourceOffset = 0; // current level stack pointer, once this reaches numSources, the level is done
    int apxFillLevel = 0; // fill level of the multipole approximation warp queue
    int bdyFillLevel = 0; // fill level of the source body warp queue

    while (numSources > 0) // While there are source cells to traverse
    {
        const int sourceIdx = sourceOffset + laneIdx;                      // Source cell index of current lane
        int sourceQueue     = cellQueue[ringAddr(oldSources + sourceIdx)]; // Global source cell index in queue
        const fvec4 MAC     = sourceCenter[sourceQueue];                   // load source cell center + MAC
        const fvec3 curSrcCenter{MAC[0], MAC[1], MAC[2]};                    // Current source cell center
        const CellData sourceData = sourceCells[sourceQueue];                // load source cell data
        const bool isNode         = sourceData.isNode();                     // Is non-leaf cell
        const bool isClose =
            applyMAC(curSrcCenter, MAC[3], sourceData, targetCenter, targetSize); // Is too close for MAC
        const bool isSource = sourceIdx < numSources;                             // Source index is within bounds

        // Split
        const bool isSplit     = isNode && isClose && isSource;        // Source cell must be split
        const int childBegin   = sourceData.child();                   // First child cell
        const int numChild     = sourceData.nchild() & -int(isSplit);  // Number of child cells (masked by split flag)
        const int numChildScan = inclusiveScanInt(numChild);           // Inclusive scan of numChild
        const int numChildLane = numChildScan - numChild;              // Exclusive scan of numChild
        const int numChildWarp = shflSync(numChildScan, GpuConfig::warpSize - 1); // Total numChild of current warp
        sourceOffset += imin(GpuConfig::warpSize, numSources - sourceOffset);     // advance current level stack pointer
        if (numChildWarp + numSources - sourceOffset > TravConfig::memPerWarp)    // If cell queue overflows
            return make_uint2(0xFFFFFFFF, 0xFFFFFFFF);                       // Exit kernel
        int childIdx = oldSources + numSources + newSources + numChildLane; // Child index of current lane
        for (int i = 0; i < numChild; i++)                                  // Loop over child cells for each lane
            cellQueue[ringAddr(childIdx + i)] = childBegin + i;             // Queue child cells for next level
        newSources += numChildWarp; //  Increment source cell count for next loop

        // Multipole approximation
        const bool isApprox = !isClose && isSource; // Source cell can be used for M2P
        int numKeepWarp     = streamCompact(&sourceQueue, isApprox, tempQueue);
        // push valid approx source cell indices into approxQueue
        const int apxTopUp = shflUpSync(sourceQueue, apxFillLevel);
        approxQueue        = (laneIdx < apxFillLevel) ? approxQueue : apxTopUp;
        apxFillLevel += numKeepWarp;

        if (apxFillLevel >= GpuConfig::warpSize) // If queue is larger than warp size,
        {
            // Call M2P kernel
            approxAcc(acc_i, pos_i, approxQueue, sourceCenter, Multipoles, EPS2, tempQueue);
            apxFillLevel -= GpuConfig::warpSize;
            // pull down remaining source cell indices into now empty approxQueue
            approxQueue = shflDownSync(sourceQueue, numKeepWarp - apxFillLevel);
            m2pCounter += warpSize;
        }

        // Direct
        const bool isLeaf       = !isNode;                               // Is leaf cell
        bool isDirect           = isClose && isLeaf && isSource;         // Source cell can be used for P2P
        const int numBodies     = sourceData.nbody() & -int(isDirect);   // Number of bodies in cell
        const int numBodiesScan = inclusiveScanInt(numBodies);           // Inclusive scan of numBodies
        int numBodiesLane       = numBodiesScan - numBodies;             // Exclusive scan of numBodies
        int numBodiesWarp       = shflSync(numBodiesScan, GpuConfig::warpSize - 1); // Total numBodies of current warp
        int prevBodyIdx         = 0;
        while (numBodiesWarp > 0) // While there are bodies to process from current source cell set
        {
            tempQueue[laneIdx] = 1; // Default scan input is 1, such that consecutive lanes load consecutive bodies
            if (isDirect && (numBodiesLane < GpuConfig::warpSize))
            {
                isDirect                 = false;                  // Set cell as processed
                tempQueue[numBodiesLane] = -1 - sourceData.body(); // Put first source cell body index into the queue
            }
            const int bodyIdx = inclusiveSegscanInt(tempQueue[laneIdx], prevBodyIdx);
            // broadcast last processed bodyIdx from the last lane to restart the scan in the next iteration
            prevBodyIdx = shflSync(bodyIdx, GpuConfig::warpSize - 1);

            if (numBodiesWarp >= GpuConfig::warpSize) // Process bodies from current set of source cells
            {
                const fvec4 sourceBody = bodyPos[bodyIdx];  // Load source body coordinates
                directAcc(sourceBody, acc_i, pos_i, EPS2);
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
                    const fvec4 sourceBody = bodyPos[bodyQueue]; // Load source body coordinates
                    directAcc(sourceBody, acc_i, pos_i, EPS2);
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
        approxAcc(acc_i, pos_i, laneIdx < apxFillLevel ? approxQueue : -1, sourceCenter, Multipoles, EPS2, tempQueue);

        m2pCounter += apxFillLevel;
    }

    if (bdyFillLevel > 0) // If there are leftover direct bodies
    {
        const int bodyIdx = laneIdx < bdyFillLevel ? bodyQueue : -1;
        // Load position of source bodies, with padding for invalid lanes
        const fvec4 sourceBody = bodyIdx >= 0 ? bodyPos[bodyIdx] : fvec4{0.0f, 0.0f, 0.0f, 0.0f};
        directAcc(sourceBody, acc_i, pos_i, EPS2);
        p2pCounter += bdyFillLevel;
    }

    return {m2pCounter, p2pCounter};
}

__device__ uint64_t sumP2PGlob     = 0;
__device__ unsigned int maxP2PGlob = 0;
__device__ uint64_t sumM2PGlob     = 0;
__device__ unsigned int maxM2PGlob = 0;

__device__ unsigned int targetCounterGlob = 0;

__global__ void resetTraversalCounters()
{
    sumP2PGlob = 0;
    maxP2PGlob = 0;
    sumM2PGlob = 0;
    maxM2PGlob = 0;

    targetCounterGlob = 0;
}

/*! @brief tree traversal
 *
 * @param[in]  firstBody     index of first body in bodyPos to compute acceleration for
 * @param[in]  lastBody      index (exclusive) of last body in @p bodyPos to compute acceleration for
 * @param[in]  images        number of periodic images to include
 * @param[in]  EPS2          Plummer softening parameter
 * @param[in]  cycle         2 * M_PI
 * @param[in]  rootRange     (start,end) index pair of cell indices to start traversal from
 * @param[in]  bodyPos       pointer to SFC-sorted bodies
 * @param[in]  srcCells      source cell data
 * @param[in]  srcCenter     source center data, x,y,z location and square of MAC radius, same order as srcCells
 * @param[in]  Multipoles    the multipole expansions in the same order as srcCells
 * @param[out] bodyAcc       body accelerations
 * @param[-]   globalPool    length proportional to number of warps in the launch grid, uninitialized
 */
__global__ __launch_bounds__(TravConfig::numThreads)
void traverse(int firstBody, int lastBody, int images, const float EPS2, float cycle,
              const int2 rootRange, const fvec4* __restrict__ bodyPos,
              const CellData* __restrict__ srcCells,
              const fvec4* __restrict__ srcCenter, const fvec4* __restrict__ Multipoles,
              fvec4* bodyAcc, int* globalPool)
{
    const int laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
    const int warpIdx = threadIdx.x >> GpuConfig::warpSizeLog2;

    constexpr int numWarpsPerBlock = TravConfig::numThreads / GpuConfig::warpSize;

    const int numTargets = (lastBody - firstBody - 1) / TravConfig::targetSize + 1;

    static_assert(NTERM <= GpuConfig::warpSize, "review approxAcc function before disabling this check");
    constexpr int smSize = (TravConfig::numThreads > NTERM * numWarpsPerBlock) ? TravConfig::numThreads : NTERM * numWarpsPerBlock;
    __shared__ int sharedPool[smSize];

    // warp-common shared mem, 1 int per thread
    int* tempQueue = sharedPool + GpuConfig::warpSize * warpIdx;
    // warp-common global mem storage
    int* cellQueue = globalPool + TravConfig::memPerWarp * ((blockIdx.x * numWarpsPerBlock) + warpIdx);

    //int targetIdx = (blockIdx.x * numWarpsPerBlock) + warpIdx;
    int targetIdx = 0;

    while (true)
    //for(; targetIdx < numTargets; targetIdx += (gridDim.x * numWarpsPerBlock))
    {
        // first thread in warp grabs next target
        if (laneIdx == 0)
        {
            // this effectively randomizes which warp gets which targets, which better balances out
            // the load imbalance between different targets compared to static assignment
            targetIdx = atomicAdd(&targetCounterGlob, 1);
        }
        targetIdx = shflSync(targetIdx, 0);

        if (targetIdx >= numTargets) return;

        const int bodyBegin = firstBody + targetIdx * TravConfig::targetSize;
        const int bodyEnd   = imin(bodyBegin + TravConfig::targetSize, lastBody);

        // load target coordinates
        fvec3 pos_i[TravConfig::nwt];
        for (int i = 0; i < TravConfig::nwt; i++)
        {
            int bodyIdx = imin(bodyBegin + i * GpuConfig::warpSize + laneIdx, bodyEnd - 1);
            pos_i[i]    = make_fvec3(fvec4(bodyPos[bodyIdx]));
        }

        fvec3 Xmin = pos_i[0];
        fvec3 Xmax = pos_i[0];
        for (int i = 1; i < TravConfig::nwt; i++)
        {
            Xmin = min(Xmin, pos_i[i]);
            Xmax = max(Xmax, pos_i[i]);
        }

        Xmin = { warpMin(Xmin[0]), warpMin(Xmin[1]), warpMin(Xmin[2]) };
        Xmax = { warpMax(Xmax[0]), warpMax(Xmax[1]), warpMax(Xmax[2]) };

        fvec3 targetCenter     = (Xmax + Xmin) * 0.5f;
        const fvec3 targetSize = (Xmax - Xmin) * 0.5f;

        fvec4 acc_i[TravConfig::nwt];
        for (int i = 0; i < TravConfig::nwt; i++)
        {
            acc_i[i] = fvec4{0, 0, 0, 0};
        }

        int numP2P = 0, numM2P = 0;
        for (int ix = -images; ix <= images; ix++)
        {
            for (int iy = -images; iy <= images; iy++)
            {
                for (int iz = -images; iz <= images; iz++)
                {
                    fvec3 Xperiodic;
                    Xperiodic[0] = ix * cycle;
                    Xperiodic[1] = iy * cycle;
                    Xperiodic[2] = iz * cycle;

                    // apply periodic shift
                    targetCenter -= Xperiodic;
                    for (int i = 0; i < TravConfig::nwt; i++)
                    {
                        pos_i[i] -= Xperiodic;
                    }

                    const uint2 counters = traverseWarp(acc_i,
                                                        pos_i,
                                                        targetCenter,
                                                        targetSize,
                                                        bodyPos,
                                                        srcCells,
                                                        srcCenter,
                                                        Multipoles,
                                                        EPS2,
                                                        rootRange,
                                                        tempQueue,
                                                        cellQueue);
                    assert(!(counters.x == 0xFFFFFFFF && counters.y == 0xFFFFFFFF));

                    // revert periodic shift
                    targetCenter += Xperiodic;
                    for (int i = 0; i < TravConfig::nwt; i++)
                    {
                        pos_i[i] += Xperiodic;
                    }

                    numM2P += counters.x;
                    numP2P += counters.y;
                }
            }
        }

        int maxP2P = numP2P;
        int sumP2P = 0;
        int maxM2P = numM2P;
        int sumM2P = 0;

        const int bodyIdx = bodyBegin + laneIdx;
        for (int i = 0; i < TravConfig::nwt; i++)
        {
            if (i * GpuConfig::warpSize + bodyIdx < bodyEnd)
            {
                sumM2P += numM2P;
                sumP2P += numP2P;
            }
        }
        #pragma unroll
        for (int i = 0; i < GpuConfig::warpSizeLog2; i++)
        {
            maxP2P = max(maxP2P, shflXorSync(maxP2P, 1 << i));
            sumP2P += shflXorSync(sumP2P, 1 << i);
            maxM2P = max(maxM2P, shflXorSync(maxM2P, 1 << i));
            sumM2P += shflXorSync(sumM2P, 1 << i);
        }
        if (laneIdx == 0)
        {
            atomicMax(&maxP2PGlob, maxP2P);
            atomicAdd((unsigned long long*)&sumP2PGlob, (unsigned long long)sumP2P);
            atomicMax(&maxM2PGlob, maxM2P);
            atomicAdd((unsigned long long*)&sumM2PGlob, (unsigned long long)sumM2P);
        }

        for (int i = 0; i < TravConfig::nwt; i++)
        {
            if (bodyIdx + i * GpuConfig::warpSize < bodyEnd)
            {
                bodyAcc[i * GpuConfig::warpSize + bodyIdx] = acc_i[i];
            }
        }
    }
}

/*! @brief Compute approximate body accelerations with Barnes-Hut
 *
 * @param[in]  firstBody     index of first body in @p bodyPos to compute acceleration for
 * @param[in]  lastBody      index (exclusive) of last body in @p bodyPos to compute acceleration for
 * @param[in]  images        number of periodic images (per direction per dimension)
 * @param[in]  eps           plummer softening parameter
 * @param[in]  cycle         2 * M_PI
 * @param[in]  bodyPos       bodies, in SFC order and as referenced by sourceCells, on device
 * @param[out] bodyAcc       output body acceleration in SFC order, on device
 * @param[in]  sourceCells   tree connectivity and body location cell data, on device
 * @param[in]  sourceCenter  center-of-mass and MAC radius^2 for each cell, on device
 * @param[in]  Multipole     cell multipoles, on device
 * @param[in]  levelRange    first and last cell of each level in the source tree, on host
 * @return                   P2P and M2P interaction statistics
 */
fvec4 computeAcceleration(int firstBody, int lastBody, int images, float eps, float cycle, const fvec4* bodyPos,
                          fvec4* bodyAcc, const CellData* sourceCells, const fvec4* sourceCenter,
                          const fvec4* Multipole, const int2* levelRange)
{
    constexpr int numWarpsPerBlock = TravConfig::numThreads / GpuConfig::warpSize;

    int numBodies = lastBody - firstBody;

    // each target gets a warp (numWarps == numTargets)
    int numWarps  = (numBodies - 1) / TravConfig::targetSize + 1;
    int numBlocks = (numWarps - 1) / numWarpsPerBlock + 1;
    numBlocks     = std::min(numBlocks, TravConfig::maxNumActiveBlocks);

    printf("launching %d blocks\n", numBlocks);

    const int poolSize = TravConfig::memPerWarp * numWarpsPerBlock * numBlocks;
    thrust::device_vector<int> globalPool(poolSize);

    resetTraversalCounters<<<1, 1>>>();
    auto t0 = std::chrono::high_resolution_clock::now();
    traverse<<<numBlocks, TravConfig::numThreads>>>(firstBody,
                                                    lastBody,
                                                    images,
                                                    eps * eps,
                                                    cycle,
                                                    {levelRange[1].x, levelRange[1].y},
                                                    bodyPos,
                                                    sourceCells,
                                                    sourceCenter,
                                                    Multipole,
                                                    bodyAcc,
                                                    rawPtr(globalPool.data()));
    kernelSuccess("traverse");

    auto t1  = std::chrono::high_resolution_clock::now();
    float dt = std::chrono::duration<float>(t1 - t0).count();

    uint64_t sumP2P, sumM2P;
    unsigned int maxP2P, maxM2P;

    checkGpuErrors(cudaMemcpyFromSymbol(&sumP2P, sumP2PGlob, sizeof(uint64_t)));
    checkGpuErrors(cudaMemcpyFromSymbol(&maxP2P, maxP2PGlob, sizeof(unsigned int)));
    checkGpuErrors(cudaMemcpyFromSymbol(&sumM2P, sumM2PGlob, sizeof(uint64_t)));
    checkGpuErrors(cudaMemcpyFromSymbol(&maxM2P, maxM2PGlob, sizeof(unsigned int)));

    fvec4 interactions;
    interactions[0] = float(sumP2P) * 1.0f / float(numBodies);
    interactions[1] = float(maxP2P);
    interactions[2] = float(sumM2P) * 1.0f / float(numBodies);
    interactions[3] = float(maxM2P);
    float flops = (interactions[0] * 20.0f + interactions[2] * 2.0f * powf(P, 3)) * float(numBodies) / dt / 1e12f;

    fprintf(stdout, "Traverse             : %.7f s (%.7f TFlops)\n", dt, flops);

    return interactions;
}

} // namespace ryoanji

