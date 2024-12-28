/*
 * Ryoanji N-body solver
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief construction of gravity data for a given octree and particle coordinates
 *
 * @author Sebastian Keller        <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/focus/source_center.hpp"
#include "cartesian_qpole.hpp"
#include "kernel.hpp"

namespace ryoanji
{

/*! @brief compute multipoles from particle data for the entire tree hierarchy
 *
 * @tparam     T1             float or double
 * @tparam     T2             float or double
 * @tparam     MType          Multipole type, e.g. CartesianQuadrupole
 * @param[in]  x              local particle x-coordinates
 * @param[in]  y              local particle y-coordinates
 * @param[in]  z              local particle z-coordinates
 * @param[in]  m              local particle masses
 * @param[in]  leafToInternal convert from a leaf index in [0:numLeafNodes] into an internal index in [0:numTreeNodes]
 * @param[in]  layout         array of length numLeafNodes + 1, layout[i] is the start offset
 *                            into the x,y,z,m arrays for the leaf node with index i. The last element
 *                            is equal to the length of the x,y,z,m arrays.
 * @param[in]  centers        expansion (com) center of each tree cell, length = numTreeNodes
 * @param[out] multipoles     output multipole moments , length = numTreeNodes
 */
template<class T1, class T2, class MType>
void computeLeafMultipoles(const T1* x, const T1* y, const T1* z, const T2* m,
                           gsl::span<const cstone::TreeNodeIndex> leafToInternal, const LocalIndex* layout,
                           const cstone::SourceCenterType<T1>* centers, MType* multipoles)
{
#pragma omp parallel for schedule(static)
    for (size_t leafIdx = 0; leafIdx < leafToInternal.size(); ++leafIdx)
    {
        TreeNodeIndex i = leafToInternal[leafIdx];
        P2M(x, y, z, m, layout[leafIdx], layout[leafIdx + 1], centers[i], multipoles[i]);
    }
}

template<class T, class MType>
void upsweepMultipoles(gsl::span<const cstone::TreeNodeIndex> levelOffset, const cstone::TreeNodeIndex* childOffsets,
                       const cstone::SourceCenterType<T>* centers, MType* multipoles)
{
    int currentLevel = levelOffset.size() - 2;

    for (; currentLevel >= 0; --currentLevel)
    {
        TreeNodeIndex start = levelOffset[currentLevel];
        TreeNodeIndex end   = levelOffset[currentLevel + 1];
#pragma omp parallel for schedule(static)
        for (TreeNodeIndex i = start; i < end; ++i)
        {
            cstone::TreeNodeIndex firstChild = childOffsets[i];
            if (firstChild)
            {
                M2M(firstChild, firstChild + cstone::eightSiblings, centers[i], centers, multipoles, multipoles[i]);
            }
        }
    }
}

} // namespace ryoanji
