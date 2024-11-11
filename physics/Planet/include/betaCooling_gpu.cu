//
// Created by Noah Kubli on 17.04.2024.
//

#include "betaCooling_gpu.hpp"
#include "sph/util/device_math.cuh"
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/findneighbors.hpp"
#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/sfc/box.hpp"

#include "sph/util/device_math.cuh"
#include "sph/particles_data.hpp"
#include "star_data.hpp"

#include <thrust/reduce.h>
#include <thrust/tuple.h>

#include <cmath>

namespace planet
{

template<typename Tpos, typename Tu, typename Ts, typename Tdu, typename Trho, typename Trho2>
__global__ void betaCoolingGPUKernel(size_t first, size_t last, const Tpos* x, const Tpos* y, const Tpos* z, Tdu* du,
                                     const Tu* u, Ts star_mass, Ts star_pos_x, Ts star_pos_y, Ts star_pos_z, Ts beta,
                                     Tpos g, const Trho* rho, Ts u_floor, Trho2 cooling_rho_limit)

{
    cstone::LocalIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= last) { return; }
    if (rho[i] >= cooling_rho_limit || u[i] <= u_floor) return;

    const double dx    = x[i] - star_pos_x;
    const double dy    = y[i] - star_pos_y;
    const double dz    = z[i] - star_pos_z;
    const double dist2 = dx * dx + dy * dy + dz * dz;
    const double dist  = sqrt(dist2);
    const double omega = sqrt(g * star_mass / (dist2 * dist));
    du[i] += -u[i] * omega / beta;
}

template<typename Dataset, typename StarData>
void betaCoolingGPU(size_t first, size_t last, Dataset& d, StarData& star)
{
    cstone::LocalIndex numParticles = last - first;
    unsigned           numThreads   = 256;
    unsigned           numBlocks    = (numParticles + numThreads - 1) / numThreads;

    betaCoolingGPUKernel<<<numBlocks, numThreads>>>(
        first, last, rawPtr(d.devData.x), rawPtr(d.devData.y), rawPtr(d.devData.z), rawPtr(d.devData.du),
        rawPtr(d.devData.u), star.m, star.position[0], star.position[1], star.position[2], star.beta, d.g,
        rawPtr(d.devData.rho), star.u_floor, star.cooling_rho_limit);

    checkGpuErrors(cudaDeviceSynchronize());
}

template void betaCoolingGPU(size_t, size_t, sphexa::ParticlesData<cstone::GpuTag>&, const StarData&);

template<typename Tu, typename Tdu>
struct AbsDivide
{
    HOST_DEVICE_FUN double operator()(const thrust::tuple<Tu, Tdu>& X)
    {
        return double{fabs(thrust::get<0>(X) / thrust::get<1>(X))};
    }
};

template<typename Dataset, typename StarData>
double duTimestepGPU(size_t first, size_t last, const Dataset& d, const StarData& star)
{
    cstone::LocalIndex numParticles = last - first;

    const auto* u  = rawPtr(d.devData.u);
    const auto* du = rawPtr(d.devData.du);

    using Tu  = std::decay_t<decltype(*u)>;
    using Tdu = std::decay_t<decltype(*du)>;

    auto begin = thrust::make_zip_iterator(u, du);
    auto end   = thrust::make_zip_iterator(u + numParticles, du + numParticles);

    double init = INFINITY;

    return star.K_u *
           thrust::transform_reduce(thrust::device, begin, end, AbsDivide<Tu, Tdu>{}, init, thrust::maximum<double>{});
}

template double duTimestepGPU(size_t, size_t, const sphexa::ParticlesData<cstone::GpuTag>&, const StarData&);

} // namespace planet
