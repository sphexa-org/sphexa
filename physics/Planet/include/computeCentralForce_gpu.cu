//
// Created by Noah Kubli on 11.03.2024.
//
#include <cub/cub.cuh>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/findneighbors.hpp"
#include "cstone/primitives/math.hpp"
#include "cstone/traversal/find_neighbors.cuh"
#include "sph/util/device_math.cuh"

#include "cstone/sfc/box.hpp"
#include "sph/particles_data.hpp"
#include "star_data.hpp"

#include "computeCentralForce_gpu.hpp"
#include "cuda_runtime.h"

namespace planet
{
template<size_t numThreads, typename Tpos, typename Ta, typename Tm, typename Tsp, typename Tsm, typename Tg,
         typename Tis, typename Tf, typename Tp>
__global__ void computeCentralForceGPUKernel(size_t first, size_t last, const Tpos* x, const Tpos* y, const Tpos* z,
                                             Ta* ax, Ta* ay, Ta* az, const Tm* m, Tsp star_pos_x, Tsp star_pos_y,
                                             Tsp star_pos_z, Tsm sm, Tg g, Tis inner_size2, Tf* star_force_block_x,
                                             Tf* star_force_block_y, Tf* star_force_block_z, Tp* star_potential_block)
{
    __shared__ Tf star_force_thread_x[numThreads];
    __shared__ Tf star_force_thread_y[numThreads];
    __shared__ Tf star_force_thread_z[numThreads];
    __shared__ Tp star_potential_thread[numThreads];

    cstone::LocalIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= last)
    {
        star_force_thread_x[threadIdx.x]   = 0.;
        star_force_thread_y[threadIdx.x]   = 0.;
        star_force_thread_z[threadIdx.x]   = 0.;
        star_potential_thread[threadIdx.x] = 0.;
    }
    else
    {
        const double dx    = x[i] - star_pos_x;
        const double dy    = y[i] - star_pos_y;
        const double dz    = z[i] - star_pos_z;
        const double dist2 = stl::max(inner_size2, dx * dx + dy * dy + dz * dz);
        const double dist  = sqrt(dist2);
        const double dist3 = dist2 * dist;

        const double a_strength = 1. / dist3 * sm * g;
        const double ax_i       = -dx * a_strength;
        const double ay_i       = -dy * a_strength;
        const double az_i       = -dz * a_strength;
        ax[i] += ax_i;
        ay[i] += ay_i;
        az[i] += az_i;

        star_force_thread_x[threadIdx.x]   = -ax_i * m[i];
        star_force_thread_y[threadIdx.x]   = -ay_i * m[i];
        star_force_thread_z[threadIdx.x]   = -az_i * m[i];
        star_potential_thread[threadIdx.x] = -g * m[i] / dist;
    }

    __syncthreads();

    for (unsigned s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (threadIdx.x < s)
        {
            star_force_thread_x[threadIdx.x] += star_force_thread_x[threadIdx.x + s];
            star_force_thread_y[threadIdx.x] += star_force_thread_y[threadIdx.x + s];
            star_force_thread_z[threadIdx.x] += star_force_thread_z[threadIdx.x + s];
            star_potential_thread[threadIdx.x] += star_potential_thread[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        star_force_block_x[blockIdx.x]   = star_force_thread_x[0];
        star_force_block_y[blockIdx.x]   = star_force_thread_y[0];
        star_force_block_z[blockIdx.x]   = star_force_thread_z[0];
        star_potential_block[blockIdx.x] = star_potential_thread[0];
    }
}

template<typename Dataset, typename StarData>
void computeCentralForceGPU(size_t first, size_t last, Dataset& d, StarData& star)
{
    cstone::LocalIndex numParticles = last - first;
    constexpr unsigned numThreads   = 256;
    unsigned           numBlocks    = (numParticles + numThreads - 1) / numThreads;

    using Tf    = typename decltype(star.force_local)::value_type;
    Tf force[3] = {};

    using Tp = std::decay_t<decltype(star.potential_local)>;
    Tp potential{0.};

    Tf* star_force_block_x;
    cudaMalloc(&star_force_block_x, sizeof(Tf) * numBlocks);
    Tf* star_force_block_y;
    cudaMalloc(&star_force_block_y, sizeof(Tf) * numBlocks);
    Tf* star_force_block_z;
    cudaMalloc(&star_force_block_z, sizeof(Tf) * numBlocks);
    Tp* star_pot_block;
    cudaMalloc(&star_pot_block, sizeof(Tp) * numBlocks);
    computeCentralForceGPUKernel<numThreads><<<numBlocks, numThreads>>>(
        first, last, rawPtr(d.devData.x), rawPtr(d.devData.y), rawPtr(d.devData.z), rawPtr(d.devData.ax),
        rawPtr(d.devData.ay), rawPtr(d.devData.az), rawPtr(d.devData.m), star.position[0], star.position[1],
        star.position[2], star.m, d.g, star.inner_size * star.inner_size, star_force_block_x, star_force_block_y,
        star_force_block_z, star_pot_block);
    checkGpuErrors(cudaGetLastError());
    checkGpuErrors(cudaDeviceSynchronize());

    force[0] =
        thrust::reduce(thrust::device, star_force_block_x, star_force_block_x + numBlocks, 0., thrust::plus<Tf>{});
    force[1] =
        thrust::reduce(thrust::device, star_force_block_y, star_force_block_y + numBlocks, 0., thrust::plus<Tf>{});
    force[2] =
        thrust::reduce(thrust::device, star_force_block_z, star_force_block_z + numBlocks, 0., thrust::plus<Tf>{});
    potential = thrust::reduce(thrust::device, star_pot_block, star_pot_block + numBlocks, 0., thrust::plus<Tp>{});

    cudaFree(star_force_block_x);
    cudaFree(star_force_block_y);
    cudaFree(star_force_block_z);
    cudaFree(star_pot_block);
    checkGpuErrors(cudaDeviceSynchronize());
}

template void computeCentralForceGPU(size_t, size_t, sphexa::ParticlesData<cstone::GpuTag>&, StarData&);
} // namespace planet
