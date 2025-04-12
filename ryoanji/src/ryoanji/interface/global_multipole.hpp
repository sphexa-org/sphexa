/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Compute global multipoles
 *
 * Pulls in both Cornerstone and Ryoanji dependencies as headers
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/focus/octree_focus_mpi.hpp"
#include "ryoanji/nbody/upsweep_cpu.hpp"

namespace ryoanji
{

template<class Tc, class Tm, class Tf, class KeyType, class MType>
void computeGlobalMultipoles(const Tc* x, const Tc* y, const Tc* z, const Tm* m, cstone::LocalIndex numParticles,
                             const cstone::Octree<KeyType>&                            globalOctree,
                             const cstone::FocusedOctree<KeyType, Tf, cstone::CpuTag>& focusTree,
                             const cstone::LocalIndex* layout, MType* multipoles)
{
    auto octree        = focusTree.octreeViewAcc();
    auto centers       = focusTree.expansionCentersAcc();
    auto globalCenters = focusTree.globalExpansionCenters();

    std::span multipoleSpan{multipoles, size_t(octree.numNodes)};
    ryoanji::computeLeafMultipoles(x, y, z, m,
                                   {octree.leafToInternal + octree.numInternalNodes, size_t(octree.numLeafNodes)},
                                   layout, centers.data(), multipoles);

    //! first upsweep with local data
    ryoanji::upsweepMultipoles({octree.levelRange, cstone::maxTreeLevel<KeyType>{} + 2}, octree.childOffsets,
                               centers.data(), multipoles);

    auto ryUpsweep = [](auto levelRange, auto childOffsets, auto M, auto centers)
    { ryoanji::upsweepMultipoles(levelRange, childOffsets.data(), centers, M); };
    cstone::globalFocusExchange(globalOctree, focusTree, multipoleSpan, ryUpsweep, globalCenters.data());

    std::vector<int, util::DefaultInitAdaptor<int>> scratch;
    focusTree.peerExchange(multipoleSpan, static_cast<int>(cstone::P2pTags::focusPeerCenters) + 1, scratch);

    //! second upsweep with leaf data from peer and global ranks in place
    ryoanji::upsweepMultipoles({octree.levelRange, cstone::maxTreeLevel<KeyType>{} + 2}, octree.childOffsets,
                               centers.data(), multipoles);
}

} // namespace ryoanji
