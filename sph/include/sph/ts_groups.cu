/*! @file
 * @brief GPU functions to manage target particle groups and block time-steps
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include "cstone/cuda/gpu_config.cuh"
#include "cstone/primitives/math.hpp"
#include "sph/sph_gpu.hpp"

namespace sph
{

using cstone::LocalIndex;

template<class T>
__global__ void groupDivvKernel(float Krho, const GroupView grp, const T* divv, float* groupDt)
{
    LocalIndex tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < grp.numGroups())
    {
        float localMax = -INFINITY;

        const LocalIndex bodyBegin = grp.groupStart(tid);
        const LocalIndex bodyEnd   = grp.groupEnd(tid);

        for (LocalIndex i = bodyBegin; i < bodyEnd; ++i)
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
    int numBlocks  = cstone::iceil(grp.numGroups(), numThreads);

    if (numBlocks == 0) { return; }
    groupDivvKernel<<<numBlocks, numThreads>>>(Krho, grp, divv, groupDt);
}

template void groupDivvTimestepGpu(float, const GroupView& grp, const float*, float*);
template void groupDivvTimestepGpu(float, const GroupView& grp, const double*, float*);

template<class T>
__global__ void groupAccKernel(float etaAcc, const GroupView grp, const T* ax, const T* ay, const T* az, const T* h,
                               float* groupDt)
{
    LocalIndex tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < grp.numGroups())
    {
        float minH2_A2 = INFINITY;

        const LocalIndex bodyBegin = grp.groupStart(tid);
        const LocalIndex bodyEnd   = grp.groupEnd(tid);

        for (LocalIndex i = bodyBegin; i < bodyEnd; ++i)
        {
            minH2_A2 = min(minH2_A2, h[i] * h[i] / norm2(cstone::Vec3<T>{ax[i], ay[i], az[i]}));
        }

        groupDt[tid] = min(groupDt[tid], etaAcc * std::sqrt(std::sqrt(minH2_A2)));
    }
}

template<class T>
void groupAccTimestepGpu(float etaAcc, const GroupView& grp, const T* ax, const T* ay, const T* az, const T* h,
                         float* groupDt)
{
    int numThreads = 256;
    int numBlocks  = cstone::iceil(grp.numGroups(), numThreads);

    if (numBlocks == 0) { return; }
    groupAccKernel<<<numBlocks, numThreads>>>(etaAcc, grp, ax, ay, az, h, groupDt);
}

template void groupAccTimestepGpu(float, const GroupView&, const double*, const double*, const double*, const double*,
                                  float*);
template void groupAccTimestepGpu(float, const GroupView&, const float*, const float*, const float*, const float*,
                                  float*);

__global__ void storeRungKernel(const GroupView grp, uint8_t rung, uint8_t* particleRungs)
{
    LocalIndex laneIdx = threadIdx.x & (cstone::GpuConfig::warpSize - 1);
    LocalIndex warpIdx = (blockDim.x * blockIdx.x + threadIdx.x) >> cstone::GpuConfig::warpSizeLog2;
    if (warpIdx >= grp.numGroups()) { return; }

    const LocalIndex bodyBegin = grp.groupStart(warpIdx);
    const LocalIndex bodyEnd   = grp.groupEnd(warpIdx);

    for (LocalIndex i = bodyBegin + laneIdx; i < bodyEnd; i += cstone::GpuConfig::warpSize)
    {
        particleRungs[i] = rung;
    }
}

void storeRungGpu(const GroupView& grp, uint8_t rung, uint8_t* particleRungs)
{
    unsigned numThreads       = 256;
    unsigned numWarpsPerBlock = numThreads / cstone::GpuConfig::warpSize;
    unsigned numBlocks        = (grp.numGroups() + numWarpsPerBlock - 1) / numWarpsPerBlock;
    if (numBlocks == 0) { return; }
    storeRungKernel<<<numBlocks, numThreads>>>(grp, rung, particleRungs);
}

template<typename T>
struct AccelerationTimestep
{
    __device__ T operator()(const thrust::tuple<T, T, T, T>& a_h)
    {
        const T               h_i = thrust::get<3>(a_h);
        const cstone::Vec3<T> A{thrust::get<0>(a_h), thrust::get<1>(a_h), thrust::get<2>(a_h)};
        return h_i * h_i / norm2(A);
    }
};

template<typename T>
T accelerationTimestepGPU(size_t first, size_t last, const T* ax, const T* ay, const T* az, const T* h)
{
    if (last <= first) { return INFINITY; }
    auto begin = thrust::make_zip_iterator(ax + first, ay + first, az + first, h + first);
    auto end   = thrust::make_zip_iterator(ax + last, ay + last, az + last, h + last);
    return thrust::transform_reduce(thrust::device, begin, end, AccelerationTimestep<T>{}, INFINITY,
                                    thrust::minimum<T>{});
}

#define ACCELERATION_TIMESTEP_GPU(T)                                                                                   \
    template T accelerationTimestepGPU(size_t first, size_t last, const T* ax, const T* ay, const T* az, const T* h);

ACCELERATION_TIMESTEP_GPU(double);
ACCELERATION_TIMESTEP_GPU(float);

} // namespace sph
