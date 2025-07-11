/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Interface for calculation of multipole moments
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/traversal/groups_gpu.h"
#include "cstone/util/reallocate.hpp"
#include "ryoanji/nbody/cartesian_qpole.hpp"
#include "ryoanji/nbody/direct.cuh"
#include "ryoanji/nbody/upsweep_gpu.h"
#include "ryoanji/nbody/upsweep_cpu.hpp"
#include "ryoanji/nbody/traversal_gpu.h"
#include "multipole_holder.cuh"

namespace ryoanji
{
using cstone::GroupView;

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType, class MType>
class MultipoleHolder<Tc, Th, Tm, Ta, Tf, KeyType, MType>::Impl
{
public:
    Impl() {}

    GroupView computeSpatialGroups(LocalIndex first, LocalIndex last, const Tc* x, const Tc* y, const Tc* z,
                                   const Th* h, const cstone::FocusedOctree<KeyType, Tf, cstone::GpuTag>& focusTree,
                                   const cstone::LocalIndex* layout, const cstone::Box<Tc>& box)
    {
        auto  d_leaves  = focusTree.treeLeavesAcc();
        float tolFactor = 2.0f;
        cstone::computeGroupSplits(first, last, x, y, z, h, d_leaves.data(), d_leaves.size() - 1, layout, box,
                                   bhMaxTargetSize(), tolFactor, traversalStack_, groups_.data);

        groups_.firstBody  = first;
        groups_.lastBody   = last;
        groups_.numGroups  = groups_.data.size() - 1;
        groups_.groupStart = rawPtr(groups_.data);
        groups_.groupEnd   = rawPtr(groups_.data) + 1;
        return groups_.view();
    }

    void upsweep(const Tc* x, const Tc* y, const Tc* z, const Tm* m, cstone::OctreeView<const KeyType> gOctree,
                 const cstone::FocusedOctree<KeyType, Tf, cstone::GpuTag>& focusTree, const cstone::LocalIndex* layout)
    {
        octree_ = focusTree.octreeViewAcc();
        resize(octree_.numLeafNodes);

        auto globalCenters = focusTree.globalExpansionCenters();

        layout_  = layout;
        centers_ = focusTree.expansionCentersAcc().data();

        computeLeafMultipoles(x, y, z, m, octree_.leafToInternal + octree_.numInternalNodes, octree_.numLeafNodes,
                              layout_, centers_, rawPtr(multipoles_));

        auto upsweepGpu = [](auto levelRange, auto childOffsets, auto M, auto centers)
        {
            int numLevels = levelRange.size() - 2;
            for (int level = numLevels - 1; level >= 0; level--)
            {
                int numCellsLevel = levelRange[level + 1] - levelRange[level];
                if (numCellsLevel)
                {
                    upsweepMultipoles(levelRange[level], levelRange[level + 1], childOffsets, centers, M);
                }
            }
        };

        //! first upsweep with local data, start at lowest possible level - 1, lowest level can only be leaves
        upsweepGpu(octree_.levelRangeSpan(), octree_.childOffsets, multipoles_.data(), centers_);

        std::span d_multipoleSpan{rawPtr(multipoles_), size_t(octree_.numNodes)};
        focusTree.globalExchange(gOctree, d_multipoleSpan, std::span<MType>{}, traversalStack_, upsweepGpu,
                                 globalCenters.data());

        focusTree.peerExchange(d_multipoleSpan, static_cast<int>(cstone::P2pTags::focusPeerCenters) + 1,
                               traversalStack_);

        //! second upsweep with leaf data from peer and global ranks in place
        upsweepGpu(octree_.levelRangeSpan(), octree_.childOffsets, multipoles_.data(), centers_);
    }

    float compute(GroupView grp, const Tc* x, const Tc* y, const Tc* z, const Tm* m, const Th* h, Tc G, int numShells,
                  const cstone::Box<Tc>& box, Ta* ugrav, Ta* ax, Ta* ay, Ta* az)
    {
        reallocate(traversalStack_, stackSize(grp.numGroups), 1.01);
        return traverse(grp, 1, x, y, z, m, h, x, y, z, m, h, octree_.childOffsets, octree_.internalToLeaf, layout_,
                        centers_, rawPtr(multipoles_), G, numShells, Vec3<Tc>{box.lx(), box.ly(), box.lz()}, ugrav, ax,
                        ay, az, (int*)rawPtr(traversalStack_));
    }

    float compute(GroupView grp, const Tc* xt, const Tc* yt, const Tc* zt, const Tm* mt, const Th* ht, const Tc* xs,
                  const Tc* ys, const Tc* zs, const Tm* ms, const Th* hs, Tc G, int numShells,
                  const cstone::Box<Tc>& box, Ta* ugrav, Ta* ax, Ta* ay, Ta* az)
    {
        reallocate(traversalStack_, stackSize(grp.numGroups), 1.01);
        return traverse(grp, 1, xt, yt, zt, mt, ht, xs, ys, zs, ms, hs, octree_.childOffsets, octree_.internalToLeaf,
                        layout_, centers_, rawPtr(multipoles_), G, numShells, Vec3<Tc>{box.lx(), box.ly(), box.lz()},
                        ugrav, ax, ay, az, (int*)rawPtr(traversalStack_));
    }

    util::array<uint64_t, 5> readStats() const { return readBhStats(); }

    const MType* deviceMultipoles() const { return rawPtr(multipoles_); }

private:
    void resize(size_t numLeaves)
    {
        double growthRate = 1.01;
        size_t numNodes   = numLeaves + (numLeaves - 1) / 7;

        reallocateDestructive(multipoles_, numNodes, growthRate);
    }

    cstone::OctreeView<const KeyType> octree_;

    //! @brief properties of focused octree nodes
    const LocalIndex*           layout_;
    const Vec4<Tf>*             centers_;
    cstone::DeviceVector<MType> multipoles_;

    //! @brief target particle group data
    cstone::GroupData<cstone::GpuTag> groups_;

    //! @brief temporary memory during traversal
    cstone::DeviceVector<LocalIndex> traversalStack_;
};

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType, class MType>
MultipoleHolder<Tc, Th, Tm, Ta, Tf, KeyType, MType>::MultipoleHolder()
    : impl_(new Impl())
{
}

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType, class MType>
MultipoleHolder<Tc, Th, Tm, Ta, Tf, KeyType, MType>::~MultipoleHolder() = default;

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType, class MType>
GroupView MultipoleHolder<Tc, Th, Tm, Ta, Tf, KeyType, MType>::computeSpatialGroups(
    LocalIndex firstBody, LocalIndex lastBody, const Tc* x, const Tc* y, const Tc* z, const Th* h,
    const cstone::FocusedOctree<KeyType, Tf, cstone::GpuTag>& focusTree, const LocalIndex* layout,
    const cstone::Box<Tc>& box)
{
    return impl_->computeSpatialGroups(firstBody, lastBody, x, y, z, h, focusTree, layout, box);
}

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType, class MType>
void MultipoleHolder<Tc, Th, Tm, Ta, Tf, KeyType, MType>::upsweep(
    const Tc* x, const Tc* y, const Tc* z, const Tm* m, cstone::OctreeView<const KeyType> gOctree,
    const cstone::FocusedOctree<KeyType, Tf, cstone::GpuTag>& focusTree, const LocalIndex* layout)
{
    impl_->upsweep(x, y, z, m, gOctree, focusTree, layout);
}

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType, class MType>
float MultipoleHolder<Tc, Th, Tm, Ta, Tf, KeyType, MType>::compute(GroupView grp, const Tc* x, const Tc* y, const Tc* z,
                                                                   const Tm* m, const Th* h, Tc G, int numShells,
                                                                   const cstone::Box<Tc>& box, Ta* ugrav, Ta* ax,
                                                                   Ta* ay, Ta* az)
{
    return impl_->compute(grp, x, y, z, m, h, G, numShells, box, ugrav, ax, ay, az);
}

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType, class MType>
float MultipoleHolder<Tc, Th, Tm, Ta, Tf, KeyType, MType>::compute(GroupView grp, const Tc* xt, const Tc* yt,
                                                                   const Tc* zt, const Tm* mt, const Th* ht,
                                                                   const Tc* xs, const Tc* ys, const Tc* zs,
                                                                   const Tm* ms, const Th* hs, Tc G, int numShells,
                                                                   const cstone::Box<Tc>& box, Ta* ugrav, Ta* ax,
                                                                   Ta* ay, Ta* az)
{
    return impl_->compute(grp, xt, yt, zt, mt, ht, xs, ys, zs, ms, hs, G, numShells, box, ugrav, ax, ay, az);
}

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType, class MType>
util::array<uint64_t, 5> MultipoleHolder<Tc, Th, Tm, Ta, Tf, KeyType, MType>::readStats() const
{
    return impl_->readStats();
}

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType, class MType>
const MType* MultipoleHolder<Tc, Th, Tm, Ta, Tf, KeyType, MType>::deviceMultipoles() const
{
    return impl_->deviceMultipoles();
}

#define MHOLDER_MTYPE(Tc, Th, Tm, Ta, Tf, KeyType, MType)                                                              \
    template class MultipoleHolder<Tc, Th, Tm, Ta, Tf, KeyType, MType>

#define MHOLDER(MType)                                                                                                 \
    MHOLDER_MTYPE(double, double, double, double, double, uint64_t, MType<double>);                                    \
    MHOLDER_MTYPE(double, float, float, float, double, uint64_t, MType<float>);                                        \
    MHOLDER_MTYPE(float, float, float, float, float, uint64_t, MType<float>);

MHOLDER(CartesianQuadrupole)
MHOLDER(CartesianMDQpole)

#define DIRECT_SUM(T)                                                                                                  \
    template void directSum(size_t, size_t, size_t, Vec3<T>, int, const T*, const T*, const T*, const T*, const T*,    \
                            T*, T*, T*, T*)

DIRECT_SUM(float);
DIRECT_SUM(double);

} // namespace ryoanji
