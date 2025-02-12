//
// Created by Noah Kubli on 17.04.2024.
//

#include "beta_cooling_gpu.hpp"
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/findneighbors.hpp"
#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/sfc/box.hpp"

#include "sph/particles_data.hpp"
#include "star_data.hpp"

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <cmath>

namespace disk
{

template<typename Treal, typename Thydro, typename Ts>
__global__ void betaCoolingGPUKernel(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z,
                                     const Treal* u, const Thydro* rho, Treal* du, Treal g, Ts star_mass,
                                     cstone::Vec3<Ts> star_position, Ts beta, Ts u_floor, Ts cooling_rho_limit)

{
    cstone::LocalIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= last) { return; }
    if (rho[i] >= cooling_rho_limit || u[i] <= u_floor) return;

    const double dx    = x[i] - star_position[0];
    const double dy    = y[i] - star_position[1];
    const double dz    = z[i] - star_position[2];
    const double dist2 = dx * dx + dy * dy + dz * dz;
    const double dist  = sqrt(dist2);
    const double omega = sqrt(g * star_mass / (dist2 * dist));
    du[i] += -u[i] * omega / beta;
}

template<typename Treal, typename Thydro>
void betaCoolingGPU(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z, const Treal* u,
                    const Thydro* rho, Treal* du, const Treal g, const StarData& star)
{
    cstone::LocalIndex numParticles = last - first;
    unsigned           numThreads   = 256;
    unsigned           numBlocks    = (numParticles + numThreads - 1) / numThreads;

    betaCoolingGPUKernel<<<numBlocks, numThreads>>>(first, last, x, y, z, u, rho, du, g, star.m, star.position,
                                                    star.beta, star.u_floor, star.cooling_rho_limit);

    checkGpuErrors(cudaDeviceSynchronize());
}

#define BETA_COOLING_GPU(Treal, Thydro)                                                                                \
    template void betaCoolingGPU(size_t first, size_t last, const Treal* x, const Treal* y, const Treal* z,            \
                                 const Treal* u, const Thydro* rho, Treal* du, const Treal g, const StarData& star);

BETA_COOLING_GPU(double, double);
BETA_COOLING_GPU(double, float);

template<typename Tu, typename Tdu>
struct AbsDivide
{
    HOST_DEVICE_FUN double operator()(const thrust::tuple<Tu, Tdu>& X)
    {
        return double{fabs(thrust::get<0>(X) / thrust::get<1>(X))};
    }
};

template<typename Treal>
double duTimestepGPU(size_t first, size_t last, const Treal* u, const Treal* du)
{
    cstone::LocalIndex numParticles = last - first;

    using Tu  = std::decay_t<decltype(*u)>;
    using Tdu = std::decay_t<decltype(*du)>;

    auto begin = thrust::make_zip_iterator(u, du);
    auto end   = thrust::make_zip_iterator(u + numParticles, du + numParticles);

    double init = INFINITY;

    return thrust::transform_reduce(thrust::device, begin, end, AbsDivide<Tu, Tdu>{}, init, thrust::minimum<double>{});
}

#define DU_TIMESTEP_GPU(Treal)                                                                                         \
    template double duTimestepGPU(size_t first, size_t last, const Treal* u, const Treal* du);

DU_TIMESTEP_GPU(double);

} // namespace disk
