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

//! @brief compute local multipoles and perform communication to build LET multipoles on each rank
template<class Tc, class Tm, class Tf, class KeyType, class MType>
void computeMultipoles(const Tc* x, const Tc* y, const Tc* z, const Tm* m, cstone::OctreeView<const KeyType> gOctree,
                       const cstone::FocusedOctree<KeyType, Tf, cstone::CpuTag>& focusTree,
                       const cstone::LocalIndex* layout, MType* multipoles)
{
    auto let           = focusTree.octreeViewAcc(); // locally essential octree
    auto centers       = focusTree.expansionCentersAcc();
    auto globalCenters = focusTree.globalExpansionCenters();

    std::span multipoleSpan{multipoles, size_t(let.numNodes)};
    ryoanji::computeLeafMultipoles(x, y, z, m, let.leafToInternalSpan(), layout, centers.data(), multipoles);

    auto upsweep = [](auto levelRange, auto childOffsets, auto M, auto centers)
    { ryoanji::upsweepMultipoles(levelRange, childOffsets, centers, M); };

    //! first upsweep with local data
    upsweep(let.levelRangeSpan(), let.childOffsets, multipoles, centers.data());

    std::vector<int, util::DefaultInitAdaptor<int>> scratch;
    focusTree.globalExchange(gOctree, multipoleSpan, std::span<MType>{}, scratch, upsweep, globalCenters.data());
    focusTree.peerExchange(multipoleSpan, static_cast<int>(cstone::P2pTags::focusPeerCenters) + 1, scratch);

    //! second upsweep with leaf data from peer and global ranks in place
    upsweep(let.levelRangeSpan(), let.childOffsets, multipoles, centers.data());
}

} // namespace ryoanji
