#include "sph/find_neighbors_gpu.hpp"
#include "sph/particles_data.hpp"

namespace sph
{

void findNeighborsSfc(const cstone::GroupView& groups, sphexa::ParticlesData<cstone::execution::Gpu>& d,
                      const cstone::Box<SphTypes::CoordinateType>& box, bool subgroups)
{
    if (d.ng0 > d.ngmax) { throw std::runtime_error("ng0 should be smaller than ngmax\n"); }

    d.neighborhood.build(groups, d, box, subgroups);
}

using cstone::GroupView;
using cstone::GpuConfig;
using cstone::LocalIndex;
using cstone::TreeNodeIndex;

__device__ bool nc_h_convergenceFailure = false;

template<class Th, class KeyType>
__global__ void updateSmoothingLengthGpuKernel(GroupView grp, unsigned ng0, const unsigned* nc, Th* h, KeyType* keys)
{
    LocalIndex laneIdx = threadIdx.x & (cstone::GpuConfig::warpSize - 1);
    LocalIndex warpIdx = (blockDim.x * blockIdx.x + threadIdx.x) >> cstone::GpuConfig::warpSizeLog2;
    if (warpIdx >= grp.numGroups) { return; }

    LocalIndex i = grp.groupStart[warpIdx] + laneIdx;
    if (i >= grp.groupEnd[warpIdx]) { return; }

    if (nc[i] <= 1)
    {
        keys[i]                 = cstone::removeKey<KeyType>{};
        nc_h_convergenceFailure = true;
    }
    h[i] = updateH(ng0, nc[i], h[i]);
}

template<class Th, class KeyType>
bool updateSmoothingLengthGpu(const GroupView& grp, unsigned ng0, const unsigned* nc, Th* h, KeyType* keys)
{
    unsigned numThreads       = 256;
    unsigned numWarpsPerBlock = numThreads / cstone::GpuConfig::warpSize;
    unsigned numBlocks        = (grp.numGroups + numWarpsPerBlock - 1) / numWarpsPerBlock;
    if (numBlocks == 0) { return false; }
    updateSmoothingLengthGpuKernel<<<numBlocks, numThreads>>>(grp, ng0, nc, h, keys);

    bool convergenceFailure;
    checkGpuErrors(cudaMemcpyFromSymbol(&convergenceFailure, GPU_SYMBOL(nc_h_convergenceFailure), sizeof(bool)));
    return convergenceFailure;
}

template bool updateSmoothingLengthGpu(const GroupView& grp, unsigned ng0, const unsigned* nc, float* h, uint64_t*);
template bool updateSmoothingLengthGpu(const GroupView& grp, unsigned ng0, const unsigned* nc, double* h, uint64_t*);

template<class Tc, class T, class KeyType>
__global__ __launch_bounds__(128) void updateSmoothingLengthIterativeGpuKernel(
    GroupView grp, unsigned ng0, unsigned ngmax, const cstone::Box<Tc> box,
    const cstone::OctreeNsView<Tc, KeyType> tree, const Tc* __restrict__ x, const Tc* __restrict__ y,
    const Tc* __restrict__ z, T* __restrict__ h, unsigned* __restrict__ nc)
{
    LocalIndex laneIdx = threadIdx.x & (cstone::GpuConfig::warpSize - 1);
    LocalIndex warpIdx = (blockDim.x * blockIdx.x + threadIdx.x) >> cstone::GpuConfig::warpSizeLog2;
    if (warpIdx >= grp.numGroups) { return; }

    const LocalIndex i = grp.groupStart[warpIdx] + laneIdx;
    if (i >= grp.groupEnd[warpIdx]) { return; }

    updateHIterative(ng0, ngmax, box, tree, i, x, y, z, h, nc);
}

template<class T, class Dataset>
void updateSmoothingLengthIterativeGpu(const cstone::GroupView& grp, Dataset& d, const cstone::Box<T>& box)
{
    unsigned numThreads       = 128;
    unsigned numWarpsPerBlock = numThreads / cstone::GpuConfig::warpSize;
    unsigned numBlocks        = (grp.numGroups + numWarpsPerBlock - 1) / numWarpsPerBlock;
    if (numBlocks == 0) { return; }

    updateSmoothingLengthIterativeGpuKernel<<<numBlocks, numThreads>>>(
        grp, d.ng0, d.ngmax, box, d.treeView, rawPtr(d.x), rawPtr(d.y), rawPtr(d.z), rawPtr(d.h), rawPtr(d.nc));
    checkGpuErrors(cudaGetLastError());
    checkGpuErrors(cudaDeviceSynchronize());
}

template void updateSmoothingLengthIterativeGpu(const cstone::GroupView&,
                                                sphexa::ParticlesData<cstone::execution::Gpu>&,
                                                const cstone::Box<SphTypes::CoordinateType>&);
} // namespace sph
