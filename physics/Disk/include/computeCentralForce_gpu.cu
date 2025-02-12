//
// Created by Noah Kubli on 11.03.2024.
//
#include <cub/cub.cuh>
#include <type_traits>
#include "cuda_runtime.h"

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/findneighbors.hpp"
#include "cstone/primitives/math.hpp"
#include "cstone/sfc/box.hpp"
#include "cstone/traversal/find_neighbors.cuh"
#include "sph/particles_data.hpp"

#include "computeCentralForce_gpu.hpp"
#include "star_data.hpp"

namespace disk
{

template<typename T>
__device__ void atomicAddVec4(cstone::Vec4<T>* x, const cstone::Vec4<T>& y)
{
    atomicAdd(&(*x)[0], y[0]);
    atomicAdd(&(*x)[1], y[1]);
    atomicAdd(&(*x)[2], y[2]);
    atomicAdd(&(*x)[3], y[3]);
}

template<size_t numThreads, typename Treal, typename Tmass, typename Ta, typename Tstar, typename Tforce>
__global__ void computeCentralForceGPUKernel(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z,
                                             Ta* ax, Ta* ay, Ta* az, const Tmass* m,
                                             const cstone::Vec3<Tstar> star_position, Tstar m_star, Treal g, Tstar inner_size2,
                                             Tforce* force_device)
{
    cstone::LocalIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;
    Tforce force{};

    if (i >= last) { force = {0., 0., 0., 0.}; }
    else
    {
        const double dx    = x[i] - star_position[0];
        const double dy    = y[i] - star_position[1];
        const double dz    = z[i] - star_position[2];
        const double dist2 = stl::max(inner_size2, dx * dx + dy * dy + dz * dz);
        const double dist  = sqrt(dist2);
        const double dist3 = dist2 * dist;

        const double a_strength = 1. / dist3 * m_star * g;
        const double ax_i       = -dx * a_strength;
        const double ay_i       = -dy * a_strength;
        const double az_i       = -dz * a_strength;
        ax[i] += ax_i;
        ay[i] += ay_i;
        az[i] += az_i;

        force[0] = -g * m[i] / dist;
        force[1] = -ax_i * m[i];
        force[2] = -ay_i * m[i];
        force[3] = -az_i * m[i];
    }

    typedef cub::BlockReduce<Tforce, numThreads>     BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    Tforce force_block = BlockReduce(temp_storage).Sum(force);
    __syncthreads();
    if (threadIdx.x == 0) { atomicAddVec4(force_device, force_block); }
}

template<typename Treal, typename Thydro, typename Tmass>
void computeCentralForceGPU(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z, Thydro* ax,
                            Thydro* ay, Thydro* az, const Tmass* m, const Treal g, StarData& star)
{
    cstone::LocalIndex numParticles = last - first;
    constexpr unsigned numThreads   = 256;
    unsigned           numBlocks    = (numParticles + numThreads - 1) / numThreads;

    star.force_local = {};
    cstone::Vec4<double>* force_device;
    checkGpuErrors(cudaMalloc(reinterpret_cast<void**>(&force_device), sizeof *force_device));
    checkGpuErrors(cudaMemcpy(force_device, &star.force_local, sizeof star.force_local, cudaMemcpyHostToDevice));

    computeCentralForceGPUKernel<numThreads><<<numBlocks, numThreads>>>(
        first, last, x, y, z, ax, ay, az, m, star.position, star.m, g, star.inner_size * star.inner_size, force_device);

    checkGpuErrors(cudaDeviceSynchronize());
    checkGpuErrors(cudaGetLastError());

    checkGpuErrors(cudaMemcpy(&star.force_local, force_device, sizeof star.force_local, cudaMemcpyDeviceToHost));
    checkGpuErrors(cudaFree(force_device));
}

#define COMPUTE_CENTRAL_FORCE_GPU(Treal, Thydro, Tmass)                                                                        \
    template void computeCentralForceGPU(size_t, size_t, const Treal* x, const Treal* y, const Treal* z, Thydro* ax,    \
                                         Thydro* ay, Thydro* az, const Tmass* m, const Treal g, StarData&);

COMPUTE_CENTRAL_FORCE_GPU(double, double, double);
COMPUTE_CENTRAL_FORCE_GPU(double, float, double);
COMPUTE_CENTRAL_FORCE_GPU(double, float, float);

} // namespace disk
