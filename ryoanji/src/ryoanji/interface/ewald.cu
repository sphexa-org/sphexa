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
__global__ void computeGravityEwaldKernel(cstone::GroupView grp, const Tc* x, const Tc* y, const Tc* z, const Tm* m,
                                          float G, Ta* ugrav, Ta* ax, Ta* ay, Ta* az,
                                          const EwaldParameters<Tc, Tmm>* ewaldParams)
{
    LocalIndex tid = blockDim.x * blockIdx.x + threadIdx.x;

    Ta Uewald = 0;
    if (tid < grp.numGroups)
    {
        for (LocalIndex i = grp.groupStart[tid]; i < grp.groupEnd[tid]; ++i)
        {
            Vec3<Tc> target{x[i], y[i], z[i]};
            Vec4<Tc> potAcc{0, 0, 0, 0};

            potAcc += computeEwaldRealSpace(target, *ewaldParams);
            potAcc += computeEwaldKSpace(target, *ewaldParams);

            Uewald += potAcc[0] * m[i];
            if (ugrav) { ugrav[i] += G * potAcc[0] * m[i]; } // potential per particle

            ax[i] += G * potAcc[1];
            ay[i] += G * potAcc[2];
            az[i] += G * potAcc[3];
        }
    }

    typedef cub::BlockReduce<Ta, EwaldKernelConfig::numThreads> BlockReduce;
    __shared__ typename BlockReduce::TempStorage                temp_storage;

    BlockReduce reduce(temp_storage);
    Ta          blockSum = reduce.Reduce(Uewald, cub::Sum());
    __syncthreads();

    if (threadIdx.x == 0) { atomicAdd(&totalEwaldPotentialGlob, blockSum); }
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

} // namespace ryoanji
