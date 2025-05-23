/*! @file
 * @brief GPU functions to manage target particle groups and block time-steps
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/cuda/gpu_config.cuh"
#include "cstone/primitives/math.hpp"
#include "sph/sph_gpu.hpp"

namespace sph
{

using cstone::LocalIndex;

template<class T>
__global__ void groupDivvKernel(float Krho, const LocalIndex* grpStart, const LocalIndex* grpEnd, LocalIndex numGroups,
                                const T* divv, float* groupDt)
{
    LocalIndex tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numGroups)
    {
        float localMax = -INFINITY;

        LocalIndex segStart = grpStart[tid];
        LocalIndex segEnd   = grpEnd[tid];

        for (LocalIndex i = segStart; i < segEnd; ++i)
        {
            localMax = max(localMax, divv[i]);
        }

        groupDt[tid] = min(groupDt[tid], Krho / abs(localMax));
    }
}

template<class T>
void groupDivvTimestepGpu(float Krho, const GroupView& grp, const T* divv, float* groupDt)
{
    int numThreads = 256;
    int numBlocks  = cstone::iceil(grp.numGroups, numThreads);

    if (numBlocks == 0) { return; }
    groupDivvKernel<<<numBlocks, numThreads>>>(Krho, grp.groupStart, grp.groupEnd, grp.numGroups, divv, groupDt);
}

template void groupDivvTimestepGpu(float, const GroupView& grp, const float*, float*);
template void groupDivvTimestepGpu(float, const GroupView& grp, const double*, float*);

template<class T>
__global__ void groupAccKernel(float etaAcc, const LocalIndex* grpStart, const LocalIndex* grpEnd, LocalIndex numGroups,
                               const T* ax, const T* ay, const T* az, float* groupDt)
{
    LocalIndex tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numGroups)
    {
        float maxAcc = 0;

        LocalIndex segStart = grpStart[tid];
        LocalIndex segEnd   = grpEnd[tid];

        for (LocalIndex i = segStart; i < segEnd; ++i)
        {
            maxAcc = max(maxAcc, norm2(cstone::Vec3<T>{ax[i], ay[i], az[i]}));
        }

        groupDt[tid] = min(groupDt[tid], etaAcc / std::sqrt(std::sqrt(maxAcc)));
    }
}

template<class T>
void groupAccTimestepGpu(float etaAcc, const GroupView& grp, const T* ax, const T* ay, const T* az, float* groupDt)
{
    int numThreads = 256;
    int numBlocks  = cstone::iceil(grp.numGroups, numThreads);

    if (numBlocks == 0) { return; }
    groupAccKernel<<<numBlocks, numThreads>>>(etaAcc, grp.groupStart, grp.groupEnd, grp.numGroups, ax, ay, az, groupDt);
}

template void groupAccTimestepGpu(float, const GroupView&, const double*, const double*, const double*, float*);
template void groupAccTimestepGpu(float, const GroupView&, const float*, const float*, const float*, float*);

__global__ void storeRungKernel(const GroupView grp, uint8_t rung, uint8_t* particleRungs)
{
    LocalIndex laneIdx = threadIdx.x & (cstone::GpuConfig::warpSize - 1);
    LocalIndex warpIdx = (blockDim.x * blockIdx.x + threadIdx.x) >> cstone::GpuConfig::warpSizeLog2;
    if (warpIdx >= grp.numGroups) { return; }

    LocalIndex i = grp.groupStart[warpIdx] + laneIdx;
    if (i >= grp.groupEnd[warpIdx]) { return; }

    particleRungs[i] = rung;
}

void storeRungGpu(const GroupView& grp, uint8_t rung, uint8_t* particleRungs)
{
    unsigned numThreads       = 256;
    unsigned numWarpsPerBlock = numThreads / cstone::GpuConfig::warpSize;
    unsigned numBlocks        = (grp.numGroups + numWarpsPerBlock - 1) / numWarpsPerBlock;
    if (numBlocks == 0) { return; }
    storeRungKernel<<<numBlocks, numThreads>>>(grp, rung, particleRungs);
}

} // namespace sph
