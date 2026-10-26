/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Ewald summation on GPUs
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/cuda/cub.hpp"
#include "cstone/traversal/groups.hpp"
#include "ryoanji/nbody/ewald.hpp"

namespace ryoanji
{

__device__ float totalEwaldPotentialGlob = 0;

struct EwaldKernelConfig
{
    //! @brief number of threads per block for the Ewald kernel
    static constexpr int numThreads = 256;
};

template<class Tc, class Ta, class Tm, class Tmm>
__device__ Ta ewaldParticle(LocalIndex i, const Tc* x, const Tc* y, const Tc* z, const Tm* m, float G, Ta* ugrav,
                            Ta* ax, Ta* ay, Ta* az, const EwaldParameters<Tc, Tmm>& ewaldParams)
{
    Vec3<Tc> target{x[i], y[i], z[i]};
    Vec4<Tc> potAcc{0, 0, 0, 0};

    potAcc += computeEwaldRealSpace(target, ewaldParams);
    potAcc += computeEwaldKSpace(target, ewaldParams);

    if (ugrav) { ugrav[i] += G * potAcc[0] * m[i]; } // potential per particle

    ax[i] += G * potAcc[1];
    ay[i] += G * potAcc[2];
    az[i] += G * potAcc[3];

    return potAcc[0] * m[i];
}

//! @brief add the per-block sum of @p Uewald to the global Ewald potential counter
template<class Ta>
__device__ void ewaldBlockReduce(Ta Uewald)
{
    typedef cub::BlockReduce<Ta, EwaldKernelConfig::numThreads> BlockReduce;
    __shared__ typename BlockReduce::TempStorage                temp_storage;

    BlockReduce reduce(temp_storage);
    Ta          blockSum = reduce.Sum(Uewald);
    __syncthreads();

    if (threadIdx.x == 0) { atomicAdd(&totalEwaldPotentialGlob, blockSum); }
}

template<class Tc, class Ta, class Tm, class Tmm>
__global__ void computeGravityEwaldKernel(cstone::GroupView grp, const Tc* x, const Tc* y, const Tc* z, const Tm* m,
                                          float G, Ta* ugrav, Ta* ax, Ta* ay, Ta* az,
                                          const EwaldParameters<Tc, Tmm>* ewaldParams)
{
    LocalIndex tid = blockDim.x * blockIdx.x + threadIdx.x;

    Ta Uewald = 0;
    if (tid < grp.numGroups)
    {
        const LocalIndex bodyBegin = grp.groupStart(tid);
        const LocalIndex bodyEnd   = grp.groupEnd(tid);
        for (LocalIndex i = bodyBegin; i < bodyEnd; ++i)
        {
            Uewald += ewaldParticle(i, x, y, z, m, G, ugrav, ax, ay, az, *ewaldParams);
        }
    }

    ewaldBlockReduce(Uewald);
}

//! @brief group-free counterpart of @p computeGravityEwaldKernel for the contiguous particle range [first:last)
template<class Tc, class Ta, class Tm, class Tmm>
__global__ void computeGravityEwaldRangeKernel(LocalIndex first, LocalIndex last, const Tc* x, const Tc* y,
                                               const Tc* z, const Tm* m, float G, Ta* ugrav, Ta* ax, Ta* ay, Ta* az,
                                               const EwaldParameters<Tc, Tmm>* ewaldParams)
{
    LocalIndex i = first + blockDim.x * blockIdx.x + threadIdx.x;

    Ta Uewald = 0;
    if (i < last)
    {
        Uewald = ewaldParticle(i, x, y, z, m, G, ugrav, ax, ay, az, *ewaldParams);
    }

    ewaldBlockReduce(Uewald);
}

__global__ void resetEwaldPotential() { totalEwaldPotentialGlob = 0; }

//! GPU version of computeGravityEwald
template<class MType, class Tc, class Ta, class Tm, class Tu>
void computeGravityEwaldGpu(const cstone::Vec3<Tc>& rootCenter, const MType& Mroot, cstone::GroupView grp, const Tc* x,
                            const Tc* y, const Tc* z, const Tm* m, const cstone::Box<Tc>& box, float G, Ta* ugrav,
                            Ta* ax, Ta* ay, Ta* az, Tu* ugravTot, EwaldSettings settings)
{
    if (box.minExtent() != box.maxExtent()) { throw std::runtime_error("Ewald gravity requires cubic bounding boxes"); }

    EwaldParameters<Tc, typename MType::value_type> ewaldParams =
        ewaldInitParameters(Mroot, rootCenter, settings.numReplicaShells, box.lx(), settings.lCut, settings.hCut,
                            settings.alpha_scale, settings.small_R_scale_factor);

    EwaldParameters<Tc, typename MType::value_type>* ewaldParamsGpu;
    checkGpuErrors(cudaMalloc(&ewaldParamsGpu, sizeof(ewaldParams)));
    checkGpuErrors(cudaMemcpy(ewaldParamsGpu, &ewaldParams, sizeof(ewaldParams), cudaMemcpyHostToDevice));

    if (ewaldParams.numEwaldShells == 0) { return; }

    unsigned numThreads = EwaldKernelConfig::numThreads;
    unsigned numBlocks  = (grp.numGroups - 1) / numThreads + 1;

    resetEwaldPotential<<<1, 1>>>();
    computeGravityEwaldKernel<<<numBlocks, numThreads>>>(grp, x, y, z, m, G, ugrav, ax, ay, az, ewaldParamsGpu);

    float totalPotential;
    checkGpuErrors(cudaMemcpyFromSymbol(&totalPotential, GPU_SYMBOL(totalEwaldPotentialGlob), sizeof(float)));
    checkGpuErrors(cudaFree(ewaldParamsGpu));

    *ugravTot += 0.5 * G * totalPotential;
}

#define COMPUTE_GRAVITY_EWALD_GPU(MType, Tc, Ta, Tm, Tu)                                                               \
    template void computeGravityEwaldGpu(const cstone::Vec3<Tc>& rootCenter, const MType& Mroot, cstone::GroupView,    \
                                         const Tc* x, const Tc* y, const Tc* z, const Tm* m,                           \
                                         const cstone::Box<Tc>& box, float G, Ta* ugrav, Ta* ax, Ta* ay, Ta* az,       \
                                         Tu* ugravTot, EwaldSettings settings)

COMPUTE_GRAVITY_EWALD_GPU(CartesianQuadrupole<double>, double, double, double, double);
COMPUTE_GRAVITY_EWALD_GPU(CartesianQuadrupole<float>, double, double, double, double);
COMPUTE_GRAVITY_EWALD_GPU(CartesianQuadrupole<float>, float, float, float, float);
COMPUTE_GRAVITY_EWALD_GPU(CartesianQuadrupole<float>, double, float, float, double);

//! GPU version of computeGravityEwald for the contiguous target range [firstTarget:lastTarget),
//! no precomputed groups required
template<class MType, class Tc, class Ta, class Tm, class Tu>
void computeGravityEwaldGpu(const cstone::Vec3<Tc>& rootCenter, const MType& Mroot, LocalIndex firstTarget,
                            LocalIndex lastTarget, const Tc* x, const Tc* y, const Tc* z, const Tm* m,
                            const cstone::Box<Tc>& box, float G, Ta* ugrav, Ta* ax, Ta* ay, Ta* az, Tu* ugravTot,
                            EwaldSettings settings)
{
    if (firstTarget >= lastTarget) { return; }

    if (box.minExtent() != box.maxExtent()) { throw std::runtime_error("Ewald gravity requires cubic bounding boxes"); }

    EwaldParameters<Tc, typename MType::value_type> ewaldParams =
        ewaldInitParameters(Mroot, rootCenter, settings.numReplicaShells, box.lx(), settings.lCut, settings.hCut,
                            settings.alpha_scale, settings.small_R_scale_factor);

    EwaldParameters<Tc, typename MType::value_type>* ewaldParamsGpu;
    checkGpuErrors(cudaMalloc(&ewaldParamsGpu, sizeof(ewaldParams)));
    checkGpuErrors(cudaMemcpy(ewaldParamsGpu, &ewaldParams, sizeof(ewaldParams), cudaMemcpyHostToDevice));

    if (ewaldParams.numEwaldShells == 0) { return; }

    unsigned numThreads = EwaldKernelConfig::numThreads;
    unsigned numBlocks  = (lastTarget - firstTarget - 1) / numThreads + 1;

    resetEwaldPotential<<<1, 1>>>();
    computeGravityEwaldRangeKernel<<<numBlocks, numThreads>>>(firstTarget, lastTarget, x, y, z, m, G, ugrav, ax, ay,
                                                              az, ewaldParamsGpu);

    float totalPotential;
    checkGpuErrors(cudaMemcpyFromSymbol(&totalPotential, GPU_SYMBOL(totalEwaldPotentialGlob), sizeof(float)));
    checkGpuErrors(cudaFree(ewaldParamsGpu));

    *ugravTot += 0.5 * G * totalPotential;
}

#define COMPUTE_GRAVITY_EWALD_RANGE_GPU(MType, Tc, Ta, Tm, Tu)                                                          \
    template void computeGravityEwaldGpu(const cstone::Vec3<Tc>& rootCenter, const MType& Mroot, LocalIndex,            \
                                         LocalIndex, const Tc* x, const Tc* y, const Tc* z, const Tm* m,                \
                                         const cstone::Box<Tc>& box, float G, Ta* ugrav, Ta* ax, Ta* ay, Ta* az,        \
                                         Tu* ugravTot, EwaldSettings settings)

COMPUTE_GRAVITY_EWALD_RANGE_GPU(CartesianQuadrupole<double>, double, double, double, double);
COMPUTE_GRAVITY_EWALD_RANGE_GPU(CartesianQuadrupole<float>, double, double, double, double);
COMPUTE_GRAVITY_EWALD_RANGE_GPU(CartesianQuadrupole<float>, float, float, float, float);
COMPUTE_GRAVITY_EWALD_RANGE_GPU(CartesianQuadrupole<float>, double, float, float, double);

} // namespace ryoanji
