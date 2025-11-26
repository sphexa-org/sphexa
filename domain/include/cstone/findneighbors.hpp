/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Find neighbors in Morton code sorted x,y,z arrays
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cmath>

#include "cstone/focus/source_center.hpp"
#include "cstone/sfc/sfc.hpp"
#include "cstone/traversal/traversal.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/util/array.hpp"

namespace cstone
{

/*! @brief compute squared distance, taking PBC into account
 *
 * Note that if pbc{X,Y,Z} is false, the result is identical to distancesq below.
 */
template<bool Pbc, class T, std::enable_if_t<Pbc, int> = 0>
HOST_DEVICE_FUN constexpr T distanceSq(T x1, T y1, T z1, T x2, T y2, T z2, const Box<T>& box)
{
    bool pbcX = (box.boundaryX() == BoundaryType::periodic);
    bool pbcY = (box.boundaryY() == BoundaryType::periodic);
    bool pbcZ = (box.boundaryZ() == BoundaryType::periodic);

    T dx = x1 - x2;
    T dy = y1 - y2;
    T dz = z1 - z2;
    // this folds d into the periodic range [-l/2, l/2] for each dimension if enabled
    dx -= pbcX * box.lx() * std::rint(dx * box.ilx());
    dy -= pbcY * box.ly() * std::rint(dy * box.ily());
    dz -= pbcZ * box.lz() * std::rint(dz * box.ilz());

    return dx * dx + dy * dy + dz * dz;
}

//! @brief compute squared distance between to points in 3D
template<bool Pbc, class T, std::enable_if_t<!Pbc, int> = 0>
HOST_DEVICE_FUN constexpr T distanceSq(T x1, T y1, T z1, T x2, T y2, T z2, const Box<T>& /*box*/)
{
    T xx = x1 - x2;
    T yy = y1 - y2;
    T zz = z1 - z2;

    return xx * xx + yy * yy + zz * zz;
}

/*! @brief findNeighbors of particle number @p id within radius
 *
 * @tparam     T               coordinate type, float or double
 * @tparam     KeyType         32- or 64-bit Morton or Hilbert key type
 * @param[in]  i               the index of the particle for which to look for neighbors
 * @param[in]  x               particle x-coordinates in SFC order (as indexed by @p tree.layout)
 * @param[in]  y               particle y-coordinates in SFC order
 * @param[in]  z               particle z-coordinates in SFC order
 * @param[in]  h               smoothing lengths (1/2 the search radius) in SFC order
 * @param[in]  tree            octree connectivity and particle indexing
 * @param[in]  box             coordinate bounding box that was used to calculate the Morton codes
 * @param[in]  ngmax           maximum number of neighbors per particle
 * @param[out] neighbors       output to store the neighbors
 * @return                     neighbor count of particle @p i, does not include self-reference; min return val is 0.
 */
template<class Tc, class Th, class KeyType>
HOST_DEVICE_FUN unsigned findNeighbors(LocalIndex i,
                                       const Tc* x,
                                       const Tc* y,
                                       const Tc* z,
                                       const Th* h,
                                       const OctreeNsView<Tc, KeyType>& tree,
                                       const Box<Tc>& box,
                                       unsigned ngmax,
                                       LocalIndex* neighbors)
{
    auto xi = x[i];
    auto yi = y[i];
    auto zi = z[i];
    auto hi = h[i];

    auto radiusSq     = Th(4.0) * hi * hi;
    auto cellRadiusSq = radiusSq * tree.searchExtFactor * tree.searchExtFactor;
    Vec3<Tc> particle{xi, yi, zi};
    unsigned numNeighbors = 0;

    auto pbc    = BoundaryType::periodic;
    bool anyPbc = box.boundaryX() == pbc || box.boundaryY() == pbc || box.boundaryZ() == pbc;
    bool usePbc = anyPbc && !insideBox(particle, {Tc(2) * hi, Tc(2) * hi, Tc(2) * hi}, box);

    auto overlapsPbc = [particle, cellRadiusSq, centers = tree.centers, sizes = tree.sizes, &box](TreeNodeIndex idx)
    { return norm2(minDistance(particle, centers[idx], sizes[idx], box)) < cellRadiusSq; };

    auto overlaps = [particle, cellRadiusSq, centers = tree.centers, sizes = tree.sizes](TreeNodeIndex idx)
    { return norm2(minDistance(particle, centers[idx], sizes[idx])) < cellRadiusSq; };

    auto searchBoxPbc =
        [i, particle, radiusSq, &tree, x, y, z, ngmax, neighbors, &numNeighbors, &box](TreeNodeIndex idx)
    {
        TreeNodeIndex leafIdx    = tree.internalToLeaf[idx];
        LocalIndex firstParticle = tree.layout[leafIdx];
        LocalIndex lastParticle  = tree.layout[leafIdx + 1];

        for (LocalIndex j = firstParticle; j < lastParticle; ++j)
        {
            if (j == i) { continue; }
            if (distanceSq<true>(x[j], y[j], z[j], particle[0], particle[1], particle[2], box) < radiusSq)
            {
                if (numNeighbors < ngmax) { neighbors[numNeighbors] = j; }
                numNeighbors++;
            }
        }
    };

    auto searchBox = [i, particle, radiusSq, &tree, x, y, z, ngmax, neighbors, &numNeighbors, &box](TreeNodeIndex idx)
    {
        TreeNodeIndex leafIdx    = tree.internalToLeaf[idx];
        LocalIndex firstParticle = tree.layout[leafIdx];
        LocalIndex lastParticle  = tree.layout[leafIdx + 1];

        for (LocalIndex j = firstParticle; j < lastParticle; ++j)
        {
            if (j == i) { continue; }
            if (distanceSq<false>(x[j], y[j], z[j], particle[0], particle[1], particle[2], box) < radiusSq)
            {
                if (numNeighbors < ngmax) { neighbors[numNeighbors] = j; }
                numNeighbors++;
            }
        }
    };

    if (usePbc) { singleTraversal(tree.childOffsets, tree.parents, overlapsPbc, searchBoxPbc); }
    else { singleTraversal(tree.childOffsets, tree.parents, overlaps, searchBox); }

    return numNeighbors;
}

template<class T, class KeyType>
void findNeighbors(const T* x,
                   const T* y,
                   const T* z,
                   const T* h,
                   LocalIndex firstId,
                   LocalIndex lastId,
                   const Box<T>& box,
                   const OctreeNsView<T, KeyType>& treeView,
                   unsigned ngmax,
                   LocalIndex* neighbors,
                   unsigned* neighborsCount)
{
    LocalIndex numWork = lastId - firstId;

#pragma omp parallel for
    for (LocalIndex i = 0; i < numWork; ++i)
    {
        LocalIndex id     = i + firstId;
        neighborsCount[i] = findNeighbors(id, x, y, z, h, treeView, box, ngmax, neighbors + i * ngmax);
    }
}

} // namespace cstone
