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

#pragma once

#include <memory>

#include "cstone/focus/octree_focus_mpi.hpp"
#include "cstone/traversal/groups.hpp"
#include "ryoanji/nbody/types.h"

namespace ryoanji
{

template<class Tc, class Th, class Tm, class Ta, class Tf, class KeyType, class MType>
class MultipoleHolder
{
public:
    MultipoleHolder();

    ~MultipoleHolder();

    cstone::GroupView computeSpatialGroups(LocalIndex first, LocalIndex last, const Tc* x, const Tc* y, const Tc* z,
                                           const Th*                                                 h,
                                           const cstone::FocusedOctree<KeyType, Tf, cstone::GpuTag>& focusTree,
                                           const cstone::LocalIndex* layout, const cstone::Box<Tc>& box);

    /*! @brief compute multipole moments with an upsweep from leaves to the tree root
     *
     * @param[in]  x, y, z, m    source particles in SFC order as referenced by layout, on GPU
     * @param[in]  globalOctree  global octree replicated across nodes on GPU
     * @param[in]  focusTree     locally essential octree focused on local domain
     * @param[in]  layout        for each leaf cell, stores the index of the first source body in cell, on GPU
     * @param[out] multipoles    output array multipoles, length=focusTree.numNodes, on host
     */
    void upsweep(const Tc* x, const Tc* y, const Tc* z, const Tm* m, const cstone::Octree<KeyType>& globalOctree,
                 const cstone::FocusedOctree<KeyType, Tf, cstone::GpuTag>& focusTree, const cstone::LocalIndex* layout,
                 MType* multipoles);

    /*! @brief compute accelerations on target particles, assuming sources = targets
     *
     * @param[in]    grp          target particle grouping into spatially compact groups
     * @param[in]    x,y,z,m,h    target and source particles in SFC order as referenced by layout, on GPU
     * @param[in]    G            gravitational constant
     * @param[in]    numShells    number of periodic images to include
     * @param[in]    box          global coordinate bounding box
     * @param[inout] ugrav        potential per particle to add to, can be nullptr, on GPU
     * @param[inout] ax, ay, az   particle accelerations to add to, on GPU
     */
    float compute(cstone::GroupView grp, const Tc* x, const Tc* y, const Tc* z, const Tm* m, const Th* h, Tc G,
                  int numShells, const cstone::Box<Tc>& box, Ta* ugrav, Ta* ax, Ta* ay, Ta* az);

    /*! @brief compute accelerations on target particles
     *
     * @param[in]    grp             target particle grouping into spatially compact groups
     * @param[in]    xt,yt,zt,mt,ht  target particles in SFC order as referenced by @p grp, on GPU
     * @param[in]    xs,ys,zs,ms,hs  source particles in SFC order as indexed by octree from last upsweep, on GPU
     * @param[in]    G               gravitational constant
     * @param[in]    numShells       number of periodic images to include
     * @param[in]    box             global coordinate bounding box
     * @param[inout] ugrav           potential per particle to add to, can be nullptr, on GPU
     * @param[inout] ax, ay, az      particle accelerations to add to, on GPU
     */
    float compute(cstone::GroupView grp, const Tc* xt, const Tc* yt, const Tc* zt, const Tm* mt, const Th* ht,
                  const Tc* xs, const Tc* ys, const Tc* zs, const Tm* ms, const Th* hs, Tc G, int numShells,
                  const cstone::Box<Tc>& box, Ta* ugrav, Ta* ax, Ta* ay, Ta* az);

    util::array<uint64_t, 5> readStats() const;

    const MType* deviceMultipoles() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

template<class T>
void directSum(size_t first, size_t last, size_t numBodies, Vec3<T> boxL, int numShells, const T* x, const T* y,
               const T* z, const T* m, const T* h, T* p, T* ax, T* ay, T* az);

} // namespace ryoanji
