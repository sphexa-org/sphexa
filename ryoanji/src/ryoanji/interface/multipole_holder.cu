#include "hip/hip_runtime.h"
/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief  Interface for calculation of multipole moments
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <thrust/device_vector.h>

#include "multipole_holder.cuh"
#include "ryoanji/nbody/cartesian_qpole.hpp"
#include "ryoanji/nbody/gpu_config.h"
#include "ryoanji/nbody/upwardpass.cuh"
#include "ryoanji/nbody/upsweep_cpu.hpp"
#include "ryoanji/nbody/traversal.cuh"

namespace ryoanji
{

template<class T>
void memcpy(T* dest, const T* src, size_t n, hipMemcpyKind kind)
{
    hipMemcpy(dest, src, sizeof(T) * n, kind);
}

template<class Tc, class Tm, class Tf, class KeyType, class MType>
class MultipoleHolder<Tc, Tm, Tf, KeyType, MType>::Impl
{
public:
    Impl() {}

    void upsweep(const Tc* x, const Tc* y, const Tc* z, const Tm* m, const cstone::Octree<KeyType>& globalOctree,
                 const cstone::FocusedOctree<KeyType, Tf>& focusTree, const cstone::LocalIndex* layout,
                 MType* multipoles)
    {
        constexpr int                  numThreads = UpsweepConfig::numThreads;
        const cstone::Octree<KeyType>& octree     = focusTree.octree();

        TreeNodeIndex numLeaves = focusTree.octree().numLeafNodes();
        resize(numLeaves);

        auto centers       = focusTree.expansionCenters();
        auto globalCenters = focusTree.globalExpansionCenters();

        // H2D leafToInternal, internalToLeaf, layout, centers, childOffsets

        const TreeNodeIndex* leafToInternal = octree.internalOrder().data();
        memcpy(rawPtr(leafToInternal_.data()), leafToInternal, numLeaves, hipMemcpyHostToDevice);

        const TreeNodeIndex* internalToLeaf = octree.toLeafOrder().data();
        memcpy(rawPtr(internalToLeaf_.data()), internalToLeaf, internalToLeaf_.size(), hipMemcpyHostToDevice);

        const TreeNodeIndex* childOffsets = octree.childOffsets().data();
        memcpy(
            rawPtr(childOffsets_.data()), octree.childOffsets().data(), childOffsets_.size(), hipMemcpyHostToDevice);

        memcpy(rawPtr(layout_.data()), layout, layout_.size(), hipMemcpyHostToDevice);
        memcpy(rawPtr(centers_.data()), centers.data(), centers.size(), hipMemcpyHostToDevice);

        computeLeafMultipoles<<<(numLeaves - 1) / numThreads + 1, numThreads>>>(x,
                                                                                y,
                                                                                z,
                                                                                m,
                                                                                rawPtr(leafToInternal_.data()),
                                                                                numLeaves,
                                                                                rawPtr(layout_.data()),
                                                                                rawPtr(centers_.data()),
                                                                                rawPtr(multipoles_.data()));

        //! first upsweep with local data
        int  numLevels  = 21;
        auto levelRange = octree.levelRange();
        for (int level = numLevels - 1; level >= 0; level--)
        {
            int numCellsLevel = levelRange[level + 1] - levelRange[level];
            int numBlocks     = (numCellsLevel - 1) / numThreads + 1;
            hipLaunchKernelGGL(upsweepMultipoles, numBlocks, numThreads, 0, 0, levelRange[level],
                                                         levelRange[level + 1],
                                                         rawPtr(childOffsets_.data()),
                                                         rawPtr(centers_.data()),
                                                         rawPtr(multipoles_.data()));
        }

        // D2H multipoles
        memcpy(multipoles, rawPtr(multipoles_.data()), multipoles_.size(), hipMemcpyDeviceToHost);

        auto ryUpsweep = [](auto levelRange, auto childOffsets, auto M, auto centers)
        { upsweepMultipoles(levelRange, childOffsets, centers, M); };

        gsl::span multipoleSpan{multipoles, size_t(octree.numTreeNodes())};
        cstone::globalFocusExchange(globalOctree, focusTree, multipoleSpan, ryUpsweep, globalCenters.data());

        focusTree.peerExchange(multipoleSpan, static_cast<int>(cstone::P2pTags::focusPeerCenters) + 1);

        // H2D multipoles
        memcpy(rawPtr(multipoles_.data()), multipoles, multipoles_.size(), hipMemcpyHostToDevice);

        //! second upsweep with leaf data from peer and global ranks in place
        for (int level = numLevels - 1; level >= 0; level--)
        {
            int numCellsLevel = levelRange[level + 1] - levelRange[level];
            int numBlocks     = (numCellsLevel - 1) / numThreads + 1;
            hipLaunchKernelGGL(upsweepMultipoles, numBlocks, numThreads, 0, 0, levelRange[level],
                                                         levelRange[level + 1],
                                                         rawPtr(childOffsets_.data()),
                                                         rawPtr(centers_.data()),
                                                         rawPtr(multipoles_.data()));
        }
    }

    float compute(LocalIndex firstBody, LocalIndex lastBody, const Tc* x, const Tc* y, const Tc* z, const Tm* m,
                  const Tm* h, Tc G, Tc* ax, Tc* ay, Tc* az)
    {
        hipLaunchKernelGGL(resetTraversalCounters, 1, 1, 0, 0);

        constexpr int numWarpsPerBlock = TravConfig::numThreads / GpuConfig::warpSize;

        LocalIndex numBodies = lastBody - firstBody;

        // each target gets a warp (numWarps == numTargets)
        int numWarps  = (numBodies - 1) / TravConfig::targetSize + 1;
        int numBlocks = (numWarps - 1) / numWarpsPerBlock + 1;
        numBlocks     = std::min(numBlocks, TravConfig::maxNumActiveBlocks);

        LocalIndex poolSize = TravConfig::memPerWarp * numWarpsPerBlock * numBlocks;

        reallocate(globalPool_, poolSize, 1.05);
        hipLaunchKernelGGL(traverse, numBlocks, TravConfig::numThreads, 0, 0, firstBody,
                                                        lastBody,
                                                        {1, 9},
                                                        x,
                                                        y,
                                                        z,
                                                        m,
                                                        h,
                                                        rawPtr(childOffsets_.data()),
                                                        rawPtr(internalToLeaf_.data()),
                                                        rawPtr(layout_.data()),
                                                        rawPtr(centers_.data()),
                                                        rawPtr(multipoles_.data()),
                                                        G,
                                                        (int*)(nullptr),
                                                        ax,
                                                        ay,
                                                        az,
                                                        rawPtr(globalPool_.data()));
        float totalPotential;
        checkGpuErrors(hipMemcpyFromSymbol(&totalPotential, HIP_SYMBOL(totalPotentialGlob), sizeof(float)));

        return 0.5f * Tc(G) * totalPotential;
    }

    const MType* deviceMultipoles() const { return rawPtr(multipoles_.data()); }

private:
    void resize(size_t numLeaves)
    {
        size_t numNodes = numLeaves + (numLeaves - 1) / 7;

        double growthRate = 1.05;
        reallocate(leafToInternal_, numLeaves, growthRate);
        reallocate(internalToLeaf_, numNodes, growthRate);
        reallocate(childOffsets_, numNodes, growthRate);

        reallocate(layout_, numLeaves + 1, growthRate);

        reallocate(centers_, numNodes, growthRate);
        reallocate(multipoles_, numNodes, growthRate);
    }

    thrust::device_vector<TreeNodeIndex> leafToInternal_;
    thrust::device_vector<TreeNodeIndex> internalToLeaf_;
    thrust::device_vector<TreeNodeIndex> childOffsets_;

    thrust::device_vector<LocalIndex> layout_;

    thrust::device_vector<Vec4<Tf>> centers_;
    thrust::device_vector<MType>    multipoles_;

    thrust::device_vector<int> globalPool_;
};

template<class Tc, class Tm, class Tf, class KeyType, class MType>
MultipoleHolder<Tc, Tm, Tf, KeyType, MType>::MultipoleHolder()
    : impl_(new Impl())
{
}

template<class Tc, class Tm, class Tf, class KeyType, class MType>
MultipoleHolder<Tc, Tm, Tf, KeyType, MType>::~MultipoleHolder() = default;

template<class Tc, class Tm, class Tf, class KeyType, class MType>
void MultipoleHolder<Tc, Tm, Tf, KeyType, MType>::upsweep(const Tc* x, const Tc* y, const Tc* z, const Tm* m,
                                                          const cstone::Octree<KeyType>&            globalTree,
                                                          const cstone::FocusedOctree<KeyType, Tf>& focusTree,
                                                          const LocalIndex* layout, MType* multipoles)
{
    impl_->upsweep(x, y, z, m, globalTree, focusTree, layout, multipoles);
}

template<class Tc, class Tm, class Tf, class KeyType, class MType>
float MultipoleHolder<Tc, Tm, Tf, KeyType, MType>::compute(LocalIndex firstBody, LocalIndex lastBody, const Tc* x,
                                                           const Tc* y, const Tc* z, const Tm* m, const Tm* h, Tc G,
                                                           Tc* ax, Tc* ay, Tc* az)
{
    return impl_->compute(firstBody, lastBody, x, y, z, m, h, G, ax, ay, az);
}

template<class Tc, class Tm, class Tf, class KeyType, class MType>
const MType* MultipoleHolder<Tc, Tm, Tf, KeyType, MType>::deviceMultipoles() const
{
    return impl_->deviceMultipoles();
}

template class MultipoleHolder<double, double, double, uint64_t, SphericalMultipole<double, 4>>;
template class MultipoleHolder<double, double, double, uint64_t, SphericalMultipole<float, 4>>;
template class MultipoleHolder<double, float, double, uint64_t, SphericalMultipole<float, 4>>;
template class MultipoleHolder<float, float, float, uint64_t, SphericalMultipole<float, 4>>;
template class MultipoleHolder<double, double, double, uint32_t, SphericalMultipole<double, 4>>;
template class MultipoleHolder<double, float, double, uint32_t, SphericalMultipole<double, 4>>;
template class MultipoleHolder<double, float, double, uint32_t, SphericalMultipole<float, 4>>;
template class MultipoleHolder<float, float, float, uint32_t, SphericalMultipole<float, 4>>;

template class MultipoleHolder<double, double, double, uint64_t, CartesianQuadrupole<double>>;
template class MultipoleHolder<double, double, double, uint64_t, CartesianQuadrupole<float>>;
template class MultipoleHolder<double, float, double, uint64_t, CartesianQuadrupole<float>>;
template class MultipoleHolder<float, float, float, uint64_t, CartesianQuadrupole<float>>;
template class MultipoleHolder<double, double, double, uint32_t, CartesianQuadrupole<double>>;
template class MultipoleHolder<double, float, double, uint32_t, CartesianQuadrupole<double>>;
template class MultipoleHolder<double, float, double, uint32_t, CartesianQuadrupole<float>>;
template class MultipoleHolder<float, float, float, uint32_t, CartesianQuadrupole<float>>;

} // namespace ryoanji
