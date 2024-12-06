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
 * @brief A wrapper to select between CPU and GPU gravity implementations
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/domain/domain.hpp"
#include "ryoanji/interface/ewald.cuh"
#include "ryoanji/interface/global_multipole.hpp"
#include "ryoanji/interface/multipole_holder.cuh"
#include "ryoanji/nbody/ewald.hpp"
#include "ryoanji/nbody/traversal_cpu.hpp"

namespace sphexa
{

template<class MType, class DomainType, class DataType>
class MultipoleHolderCpu
{
    using Ta = typename std::decay_t<decltype(DataType{}.ax)>::value_type;
    using Tu = typename std::decay_t<decltype(DataType{}.x)>::value_type;

public:
    MultipoleHolderCpu() = default;

    cstone::GroupView computeSpatialGroups(const DataType& /*d*/, const DomainType& domain)
    {
        return {.firstBody = domain.startIndex(), .lastBody = domain.endIndex()};
    }

    void upsweep(const DataType& d, const DomainType& domain)
    {
        //! includes tree plus associated information, like peer ranks, assignment, counts, centers, etc
        const auto& focusTree = domain.focusTree();

        reallocate(multipoles_, focusTree.octreeViewAcc().numNodes, 1.05);
        ryoanji::computeGlobalMultipoles(d.x.data(), d.y.data(), d.z.data(), d.m.data(), d.x.size(),
                                         domain.globalTree(), domain.focusTree(), domain.layout().data(),
                                         multipoles_.data());
    }

    void traverse(cstone::GroupView /*grp*/, DataType& d, const DomainType& domain)
    {
        //! includes tree plus associated information, like peer ranks, assignment, counts, centers, etc
        const auto& focusTree = domain.focusTree();
        //! the focused octree, structure only
        const auto octree = focusTree.octreeViewAcc();

        const auto& box       = domain.box();
        bool        usePbc    = box.boundaryX() == cstone::BoundaryType::periodic;
        int         numShells = usePbc ? ewaldSettings_.numReplicaShells : 0;

        d.egrav         = 0;
        auto* potential = d.potential.empty() ? nullptr : d.potential.data();

        ryoanji::computeGravity(octree.childOffsets, octree.internalToLeaf, focusTree.expansionCentersAcc().data(),
                                multipoles_.data(), domain.layout().data(), domain.startCell(), domain.endCell(),
                                d.x.data(), d.y.data(), d.z.data(), d.h.data(), d.m.data(), domain.box(), d.g,
                                potential, d.ax.data(), d.ay.data(), d.az.data(), &d.egrav, numShells);

        if (usePbc)
        {
            ryoanji::computeGravityEwald(makeVec3(focusTree.expansionCentersAcc()[0]), multipoles_.front(),
                                         domain.startIndex(), domain.endIndex(), d.x.data(), d.y.data(), d.z.data(),
                                         d.m.data(), box, d.g, d.potential.data(), d.ax.data(), d.ay.data(),
                                         d.az.data(), &d.egrav, ewaldSettings_);
        }
    }

    util::array<uint64_t, 5> readStats() const { return {0, 0, 0, 0, 0}; }

    const MType* multipoles() const { return multipoles_.data(); }

private:
    std::vector<MType>     multipoles_;
    ryoanji::EwaldSettings ewaldSettings_;
};

template<class MType, class DomainType, class DataType>
class MultipoleHolderGpu
{
    using KeyType = typename DataType::KeyType;
    using Tc      = typename std::decay_t<decltype(DataType{}.x)>::value_type;
    using Th      = typename std::decay_t<decltype(DataType{}.h)>::value_type;
    using Tm      = typename std::decay_t<decltype(DataType{}.m)>::value_type;
    using Ta      = typename std::decay_t<decltype(DataType{}.ax)>::value_type;
    using Tf      = typename DomainType::RealType;

public:
    MultipoleHolderGpu() = default;

    cstone::GroupView computeSpatialGroups(const DataType& d, const DomainType& domain)
    {
        return mHolder_.computeSpatialGroups(domain.startIndex(), domain.endIndex(), rawPtr(d.devData.x),
                                             rawPtr(d.devData.y), rawPtr(d.devData.z), rawPtr(d.devData.h),
                                             domain.focusTree(), domain.layout().data(), domain.box());
    }

    void upsweep(const DataType& d, const DomainType& domain)
    {
        const auto& focusTree = domain.focusTree();
        reallocate(multipoles_, focusTree.octreeViewAcc().numNodes, 1.05);
        mHolder_.upsweep(rawPtr(d.devData.x), rawPtr(d.devData.y), rawPtr(d.devData.z), rawPtr(d.devData.m),
                         domain.globalTree(), focusTree, domain.layout().data(), multipoles_.data());
    }

    void traverse(cstone::GroupView grp, DataType& d, const DomainType& domain)
    {
        const auto& box       = domain.box();
        bool        usePbc    = box.boundaryX() == cstone::BoundaryType::periodic;
        int         numShells = usePbc ? ewaldSettings_.numReplicaShells : 0;

        d.egrav =
            mHolder_.compute(grp, rawPtr(d.devData.x), rawPtr(d.devData.y), rawPtr(d.devData.z), rawPtr(d.devData.m),
                             rawPtr(d.devData.h), d.g, numShells, domain.box(), rawPtr(d.devData.potential),
                             rawPtr(d.devData.ax), rawPtr(d.devData.ay), rawPtr(d.devData.az));

        auto stats = mHolder_.readStats();

        auto maxP2P = stats[1];
        if (maxP2P == 0xFFFFFFFF) { throw std::runtime_error("GPU traversal stack exhausted in Barnes-Hut\n"); }

        if (usePbc)
        {
            ryoanji::Vec4<Tf> rootCenter;
            memcpyD2H(domain.focusTree().expansionCentersAcc().data(), 1, &rootCenter);
            MType rootM;
            memcpyD2H(mHolder_.deviceMultipoles(), 1, &rootM);

            computeGravityEwaldGpu(makeVec3(rootCenter), rootM, grp, rawPtr(d.devData.x), rawPtr(d.devData.y),
                                   rawPtr(d.devData.z), rawPtr(d.devData.m), box, d.g, rawPtr(d.devData.potential),
                                   rawPtr(d.devData.ax), rawPtr(d.devData.ay), rawPtr(d.devData.az), &d.egrav,
                                   ewaldSettings_);
        }

        d.devData.stackUsedGravity = stats[4];
    }

    //! @brief return numP2P, maxP2P, numM2P, maxM2P, maxStack stats
    util::array<uint64_t, 5> readStats() const { return mHolder_.readStats(); }

    const MType* multipoles() const { return multipoles_.data(); }

private:
    ryoanji::MultipoleHolder<Tc, Th, Tm, Ta, Tf, KeyType, MType> mHolder_;
    std::vector<MType>                                           multipoles_;
    ryoanji::EwaldSettings                                       ewaldSettings_;
};

} // namespace sphexa
