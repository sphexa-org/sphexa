/*! @file
 * @brief Polytropic equation of state
 *
 * @author Noah Kubli <noah.kubli@uzh.ch>
 */

#include "cstone/primitives/math.hpp"
#include "cstone/util/tuple.hpp"
#include "eos_polytropic_gpu.hpp"
#include "eos_polytropic_loop.hpp"
#include "sph/particles_data.hpp"
#include "star_data.hpp"

namespace planet
{

template<typename T1, typename T2, typename T3, typename Trho, typename Tp, typename Tc>
__global__ void computePolytropicEOS_HydroStdKernel(size_t firstParticle, size_t lastParticle, T1 Kpoly, T2 exp_poly,
                                                    T3 gamma, const Trho* rho, Tp* p, Tc* c)
{
    unsigned i = firstParticle + blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= lastParticle) return;

    util::tie(p[i], c[i]) = polytropicEOS(Kpoly, exp_poly, gamma, rho[i]);
}

template<typename Dataset, typename StarData>
void computePolytropicEOS_HydroStdGPU(size_t firstParticle, size_t lastParticle, Dataset& d, const StarData& star)
{
    if (firstParticle == lastParticle) { return; }
    unsigned numThreads = 256;
    unsigned numBlocks  = cstone::iceil(lastParticle - firstParticle, numThreads);
    computePolytropicEOS_HydroStdKernel<<<numBlocks, numThreads>>>(firstParticle, lastParticle, star.Kpoly,
                                                                   star.exp_poly, d.gamma, rawPtr(d.devData.rho),
                                                                   rawPtr(d.devData.p), rawPtr(d.devData.c));

    checkGpuErrors(cudaDeviceSynchronize());
}

template void computePolytropicEOS_HydroStdGPU(size_t, size_t, sphexa::ParticlesData<cstone::GpuTag>&, const StarData&);
} // namespace planet
