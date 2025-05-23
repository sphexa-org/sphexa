//
// Created by Noah Kubli on 12.03.2024.
//

#include "cstone/cuda/cub.hpp"
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/tree/definitions.h"

#include "accretion_gpu.hpp"
#include "star_data.hpp"
#include "removal_statistics.hpp"

namespace disk
{

__device__ void atomicAddRS(RemovalStatistics* x, const RemovalStatistics& y)
{
    atomicAdd(&(x->mass), y.mass);
    atomicAdd(&(x->momentum[0]), y.momentum[0]);
    atomicAdd(&(x->momentum[1]), y.momentum[1]);
    atomicAdd(&(x->momentum[2]), y.momentum[2]);
    atomicAdd(&(x->count), y.count);
}

template<typename Tkeys, typename Tm, typename Tv>
__device__ void markForRemovalAndAdd(RemovalStatistics& statistics, size_t i, Tkeys* keys, const Tm* m, const Tv* vx,
                                     const Tv* vy, const Tv* vz)
{
    keys[i]                = cstone::removeKey<Tkeys>::value;
    statistics.mass        = m[i];
    statistics.momentum[0] = m[i] * vx[i];
    statistics.momentum[1] = m[i] * vy[i];
    statistics.momentum[2] = m[i] * vz[i];
    statistics.count       = 1;
}

template<unsigned numThreads, typename T1, typename Th, typename Tkeys, typename T2, typename Tm, typename Tv>
__global__ void computeAccretionConditionKernel(size_t first, size_t last, const T1* x, const T1* y, const T1* z,
                                                const Th* h, Tkeys* keys, const Tm* m, const Tv* vx, const Tv* vy,
                                                const Tv* vz, const cstone::Vec3<T2> star_position, T2 star_size2,
                                                T2 removal_limit_h, RemovalStatistics* device_accreted,
                                                RemovalStatistics* device_removed)
{
    cstone::LocalIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;

    // Accreted particles statistics
    RemovalStatistics accreted{};
    // Removed particles statistics
    RemovalStatistics removed{};

    if (i >= last) {}
    else
    {
        const double dx    = x[i] - star_position[0];
        const double dy    = y[i] - star_position[1];
        const double dz    = z[i] - star_position[2];
        const double dist2 = dx * dx + dy * dy + dz * dz;

        if (dist2 < star_size2) { markForRemovalAndAdd(accreted, i, keys, m, vx, vy, vz); }
        else if (h[i] > removal_limit_h) { markForRemovalAndAdd(removed, i, keys, m, vx, vy, vz); }
    }

    typedef cub::BlockReduce<RemovalStatistics, numThreads> BlockReduce;
    __shared__ typename BlockReduce::TempStorage            temp_storage;

    RemovalStatistics block_accreted = BlockReduce(temp_storage).Sum(accreted);
    __syncthreads();
    if (threadIdx.x == 0) { atomicAddRS(device_accreted, block_accreted); }

    RemovalStatistics block_removed = BlockReduce(temp_storage).Sum(removed);
    __syncthreads();
    if (threadIdx.x == 0) { atomicAddRS(device_removed, block_removed); }
}

template<typename Treal, typename Thydro, typename Tkeys, typename Tmass>
void computeAccretionConditionGPU(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z,
                                  const Thydro* h, Tkeys* keys, const Tmass* m, const Thydro* vx, const Thydro* vy,
                                  const Thydro* vz, StarData& star)
{
    cstone::LocalIndex numParticles = last - first;
    constexpr unsigned numThreads   = 256;
    unsigned           numBlocks    = (numParticles + numThreads - 1) / numThreads;

    star.accreted_local = {};
    star.removed_local  = {};

    RemovalStatistics *accreted_device, *removed_device;
    checkGpuErrors(cudaMalloc(reinterpret_cast<void**>(&accreted_device), sizeof *accreted_device));
    checkGpuErrors(cudaMalloc(reinterpret_cast<void**>(&removed_device), sizeof *removed_device));
    checkGpuErrors(
        cudaMemcpy(accreted_device, &star.accreted_local, sizeof star.accreted_local, cudaMemcpyHostToDevice));
    checkGpuErrors(cudaMemcpy(removed_device, &star.removed_local, sizeof star.removed_local, cudaMemcpyHostToDevice));

    computeAccretionConditionKernel<numThreads><<<numBlocks, numThreads>>>(
        first, last, x, y, z, h, keys, m, vx, vy, vz, star.position, star.inner_size * star.inner_size,
        star.removal_limit_h, accreted_device, removed_device);

    checkGpuErrors(cudaDeviceSynchronize());
    checkGpuErrors(cudaGetLastError());

    checkGpuErrors(
        cudaMemcpy(&star.accreted_local, accreted_device, sizeof star.accreted_local, cudaMemcpyDeviceToHost));
    checkGpuErrors(cudaMemcpy(&star.removed_local, removed_device, sizeof star.removed_local, cudaMemcpyDeviceToHost));
    checkGpuErrors(cudaFree(accreted_device));
    checkGpuErrors(cudaFree(removed_device));
}

#define COMPUTE_ACCRETION_CONDITION_GPU(Treal, Thydro, Tkeys, Tmass)                                                   \
    template void computeAccretionConditionGPU(size_t first, size_t last, const Treal* x, const Treal* y,              \
                                               const Treal* z, const Thydro* h, Tkeys* keys, const Tmass* m,           \
                                               const Thydro* vx, const Thydro* vy, const Thydro* vz, StarData& star);

COMPUTE_ACCRETION_CONDITION_GPU(double, double, size_t, double);
COMPUTE_ACCRETION_CONDITION_GPU(double, float, size_t, double);
COMPUTE_ACCRETION_CONDITION_GPU(double, float, size_t, float);

} // namespace disk
