//
// Created by Noah Kubli on 11.03.2024.
//
#include "cstone/cuda/cub.hpp"
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/primitives/math.hpp"

#include "central_force_gpu.hpp"
#include "central_force_loop.hpp"
#include "star_data.hpp"

namespace disk
{

static __device__ cstone::Vec4<double> force_device;

template<typename T>
__device__ void atomicAddVec4(cstone::Vec4<T>* x, const cstone::Vec4<T>& y)
{
    atomicAdd(&(*x)[0], y[0]);
    atomicAdd(&(*x)[1], y[1]);
    atomicAdd(&(*x)[2], y[2]);
    atomicAdd(&(*x)[3], y[3]);
}

// template<size_t numThreads, typename Treal, typename Tmass, typename Ta, typename Tstar>
template<size_t numThreads, typename Data>
__global__ void computeCentralForceGPUKernel(size_t first, size_t last, const Data d, /*const Treal* x, const Treal* y,
                                              const Treal* z, Ta* ax, Ta* ay, Ta* az, const Tmass* m, Treal g,
                                              cstone::Vec3<Tstar> star_position, Tstar m_star, Tstar inner_size2,*/
                                             StarPotentialType potentialType)
{
    cstone::LocalIndex   i = first + blockDim.x * blockIdx.x + threadIdx.x;
    cstone::Vec4<double> force{0., 0., 0., 0.};

    //    const CentralForceData data{
    //        d.x, d.y, d.z, d.m, d.ax, d.ay, d.az, d.g, star.position, star.m, star.inner_size * star.inner_size, 1.0};

    if (i >= last) { force = {0., 0., 0., 0.}; }
    else
    {
        if (potentialType == StarPotentialType::newtonian) { newtonianGravity(d, i, force); }
        else if (potentialType == StarPotentialType::einstein_precession) { einsteinPrecession(d, i, force); }
        //        const double dx    = x[i] - star_position[0];
        //        const double dy    = y[i] - star_position[1];
        //        const double dz    = z[i] - star_position[2];
        //        const double dist2 = stl::max(inner_size2, dx * dx + dy * dy + dz * dz);
        //        const double dist  = sqrt(dist2);
        //        const double dist3 = dist2 * dist;
        //
        //        const double a_strength = 1. / dist3 * m_star * g;
        //        const double ax_i       = -dx * a_strength;
        //        const double ay_i       = -dy * a_strength;
        //        const double az_i       = -dz * a_strength;
        //        ax[i] += ax_i;
        //        ay[i] += ay_i;
        //        az[i] += az_i;
        //
        //        force[0] = -g * m[i] / dist;
        //        force[1] = -ax_i * m[i];
        //        force[2] = -ay_i * m[i];
        //        force[3] = -az_i * m[i];
    }

    typedef cub::BlockReduce<cstone::Vec4<double>, numThreads> BlockReduce;
    __shared__ typename BlockReduce::TempStorage               temp_storage;

    cstone::Vec4<double> force_block = BlockReduce(temp_storage).Sum(force);
    __syncthreads();
    if (threadIdx.x == 0) { atomicAddVec4(&force_device, force_block); }
}

template<typename Treal, typename Thydro, typename Tmass>
void computeCentralForceGPU(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z, Thydro* ax,
                            Thydro* ay, Thydro* az, const Tmass* m, Treal g, StarData& star)
{
    cstone::LocalIndex numParticles = last - first;
    constexpr unsigned numThreads   = 256;
    unsigned           numBlocks    = (numParticles + numThreads - 1) / numThreads;

    cstone::Vec4<double> force_local{0., 0., 0., 0.};
    checkGpuErrors(cudaMemcpyToSymbol(GPU_SYMBOL(force_device), &force_local, sizeof(force_local)));

    const double     inner_size2 = star.inner_size * star.inner_size;
    CentralForceData data{x, y, z, m, ax, ay, az, g, star.m, inner_size2, 1.0};
    data.star_position = star.position; // Initializing in aggregate list produces an error in CUDA compiler

    computeCentralForceGPUKernel<numThreads><<<numBlocks, numThreads>>>(first, last, data, star.potentialType);

    checkGpuErrors(cudaDeviceSynchronize());
    checkGpuErrors(cudaGetLastError());

    checkGpuErrors(cudaMemcpyFromSymbol(&force_local, GPU_SYMBOL(force_device), sizeof(force_local)));
    star.force_local = force_local;
}

#define COMPUTE_CENTRAL_FORCE_GPU(Treal, Thydro, Tmass)                                                                \
    template void computeCentralForceGPU(size_t, size_t, const Treal* x, const Treal* y, const Treal* z, Thydro* ax,   \
                                         Thydro* ay, Thydro* az, const Tmass* m, Treal g, StarData&);

COMPUTE_CENTRAL_FORCE_GPU(double, double, double);
COMPUTE_CENTRAL_FORCE_GPU(double, float, double);
COMPUTE_CENTRAL_FORCE_GPU(double, float, float);

} // namespace disk
