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
 * @brief  Compute global multipoles
 *
 * Pulls in both Cornerstone and Ryoanji dependencies as headers
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <memory>

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

    gsl::span multipoleSpan{multipoles, size_t(octree.numNodes)};
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
