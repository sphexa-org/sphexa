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
 * @brief Barnes-Hut tree walk to compute gravity forces on particles in leaf cells
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/traversal/traversal.hpp"
#include "cstone/traversal/macs.hpp"
#include "cstone/focus/source_center.hpp"
#include "cartesian_qpole.hpp"
#include "kernel.hpp"

namespace ryoanji
{

//! @brief compute geometric center and size of the bounding box of particles in the given range
template<class T, size_t N>
auto computeCenterAndSize(const util::array<Vec4<T>, N>& target)
{
    Vec3<T> tMin = makeVec3(target[0]);
    Vec3<T> tMax = tMin;
    for (LocalIndex i = 1; i < N; ++i)
    {
        Vec3<T> tp = makeVec3(target[i]);
        tMin       = min(tp, tMin);
        tMax       = max(tp, tMax);
    }

    Vec3<T> center = (tMax + tMin) * T(0.5);
    Vec3<T> size   = (tMax - tMin) * T(0.5);

    return std::make_tuple(center, size);
}

/*! @brief computes gravitational acceleration for all particles in the specified group
 *
 * @tparam T1                   float or double
 * @tparam T2                   float or double
 * @param[in]    target         (x,y,z,h) of N target particles
 * @param[in]    childOffsets   child node index of each node
 * @param[in]    internalToLeaf map to convert an octree node index into a cstone leaf index
 * @param[in]    centers        (x,y,z,mac^2) expansion center for each tree cell
 * @param[in]    multipoles     array of length @p octree.numTreeNodes() with the multipole moments for all nodes
 * @param[in]    layout         array of length @p octree.numLeafNodes()+1, layout[i] is the start offset
 *                              into the x,y,z,m arrays for the leaf node with index i. The last element
 *                              is equal to the length of the x,y,z,m arrays.
 * @param[in]    x              x-coordinates (sources)
 * @param[in]    y              y-coordinates
 * @param[in]    z              z-coordinates
 * @param[in]    h              smoothing lengths
 * @param[in]    m              masses
 * @param[inout] acc            acceleration and potential of N target particles to add to
 *
 * Note: acceleration output is added to destination
 */
template<class MType, class T1, class Th, class Tm, size_t N>
void computeGravityGroup(const util::array<Vec4<T1>, N>& target, const TreeNodeIndex* childOffsets,
                         const TreeNodeIndex* internalToLeaf, const cstone::SourceCenterType<T1>* centers,
                         MType* multipoles, const LocalIndex* layout, const T1* x, const T1* y, const T1* z,
                         const Th* h, const Tm* m, Vec4<T1>* acc)
{
    Vec3<T1> targetCenter, targetSize;
    std::tie(targetCenter, targetSize) = computeCenterAndSize(target);

    /*! @brief octree traversal continuation criterion
     *
     * This functor takes an octree node index as argument. It then evaluates the MAC between
     * the targetBox and the argument octree node. Returns true if the MAC failed to signal
     * the traversal routine to keep going. If the MAC passed, the multipole moments are applied
     * to the particles in the target box and traversal is stopped.
     */
    auto descendOrM2P = [centers, multipoles, &target, &targetCenter, &targetSize, acc](TreeNodeIndex idx)
    {
        const auto& com = centers[idx];
        const auto& mp  = multipoles[idx];

        bool violatesMac = cstone::evaluateMac(makeVec3(com), com[3], targetCenter, targetSize);

        if (!violatesMac)
        {
// apply multipole to all particles in group
#if defined(__llvm__) || defined(__clang__)
#pragma clang loop vectorize(enable)
#endif
            for (LocalIndex k = 0; k < N; ++k)
            {
                acc[k] = M2P(acc[k], makeVec3(target[k]), makeVec3(com), mp);
            }
        }

        return violatesMac;
    };

    /*! @brief traversal endpoint action
     *
     * This functor gets called with an octree leaf node index whenever traversal hits a leaf node
     * and the leaf failed the MAC w.r.t to the target box. In that case, direct particle-particle
     * interactions need to be computed.
     */
    auto leafP2P = [internalToLeaf, layout, &target, x, y, z, h, m, acc](TreeNodeIndex idx)
    {
        TreeNodeIndex lidx        = internalToLeaf[idx];
        LocalIndex    firstSource = layout[lidx];
        LocalIndex    lastSource  = layout[lidx + 1];

        // loop over targets in group
        for (LocalIndex k = 0; k < N; ++k)
        {
            // loop over sources in cell
            for (LocalIndex s = firstSource; s < lastSource; ++s)
            {
                acc[k] = P2P(acc[k], makeVec3(target[k]), Vec3<T1>{x[s], y[s], z[s]}, m[s], Th(target[k][3]), h[s]);
            }
        }
    };

    cstone::singleTraversal(childOffsets, descendOrM2P, leafP2P);
}

/*! @brief repeats computeGravityGroup for all leaf node indices specified
 *
 * only computes total gravitational energy, no potential per particle
 *
 * @tparam       MType           Multipole type including expansion order, e.g. spherical or cartesian
 * @param[in]    childOffsets    child node index of each node
 * @param[in]    internalToLeaf  map to convert an octree node index into a cstone leaf index
 * @param[in]    macSpheres      (x,y,z,mac^2) expansion center for each tree cell
 * @param[in]    multipoles      array of length @p octree.numTreeNodes() with the multipole moments for all nodes
 * @param[in]    layout          array of length @p octree.numLeafNodes()+1, layout[i] is the start offset
 *                               into the x,y,z,m arrays for the leaf node with index i. The last element
 *                               is equal to the length of the x,y,z,m arrays.
 * @param[in]    firstLeafIndex
 * @param[in]    lastLeafIndex
 * @param[in]    x               x-coordinates
 * @param[in]    y               y-coordinates
 * @param[in]    z               z-coordinates
 * @param[in]    h               smoothing lengths
 * @param[in]    m               masses
 * @param[in]    box             global coordinate bounding box
 * @param[in]    G               gravitational constant
 * @param[inout] ugrav           location to add gravitational potential to, per particle (ignored if NULL)
 * @param[inout] ax              location to add x-acceleration to, per particle
 * @param[inout] ay              location to add y-acceleration to, per particle
 * @param[inout] az              location to add z-acceleration to, per particle
 * @param[inout] ugravTot        total gravitational potential, one element
 * @param[in]    numShells       number of periodic images to include per dimension
 */
template<class MType, class T1, class T2, class Tm>
void computeGravity(const TreeNodeIndex* childOffsets, const TreeNodeIndex* internalToLeaf,
                    const cstone::SourceCenterType<T1>* macSpheres, const MType* multipoles, const LocalIndex* layout,
                    TreeNodeIndex firstLeafIndex, TreeNodeIndex lastLeafIndex, const T1* x, const T1* y, const T1* z,
                    const T2* h, const Tm* m, const cstone::Box<T1>& box, float G, T2* ugrav, T2* ax, T2* ay, T2* az,
                    T1* ugravTot, int numShells = 0)
{
    constexpr LocalIndex groupSize   = 16;
    LocalIndex           firstTarget = layout[firstLeafIndex];
    LocalIndex           lastTarget  = layout[lastLeafIndex];

    T1 ugravLoc = 0.0;

#pragma omp parallel for reduction(+ : ugravLoc)
    for (LocalIndex i = firstTarget; i < lastTarget; i += groupSize)
    {
        util::array<Vec4<T1>, groupSize> targets, potAndAcc;

        LocalIndex groupSizeValid = std::min(groupSize, lastTarget - i);
        for (LocalIndex k = 0; k < groupSizeValid; ++k)
        {
            targets[k]   = {x[i + k], y[i + k], z[i + k], T1(h[i + k])};
            potAndAcc[k] = {0, 0, 0, 0};
        }

        for (int iz = -numShells; iz <= numShells; ++iz)
        {
            for (int iy = -numShells; iy <= numShells; ++iy)
            {
                for (int ix = -numShells; ix <= numShells; ++ix)
                {
                    Vec4<T1> pbcShift{ix * box.lx(), iy * box.ly(), iz * box.lz(), 0};

                    auto targetsShifted = targets;
                    for (auto& t_ : targetsShifted)
                    {
                        t_ -= pbcShift;
                    }

                    computeGravityGroup(targetsShifted, childOffsets, internalToLeaf, macSpheres, multipoles, layout, x,
                                        y, z, h, m, potAndAcc.data());
                }
            }
        }

        for (LocalIndex k = 0; k < groupSizeValid; ++k)
        {
            auto u = G * m[i + k] * potAndAcc[k][0];
            ugravLoc += u;
            if (ugrav) { ugrav[i + k] += u; }
            ax[i + k] += G * potAndAcc[k][1];
            ay[i + k] += G * potAndAcc[k][2];
            az[i + k] += G * potAndAcc[k][3];
        }
    }

    *ugravTot += 0.5 * ugravLoc;
}

//! @brief compute direct gravity sum for all particles [0:numParticles]
template<class Tc, class Th, class Tm>
void directSum(const Tc* x, const Tc* y, const Tc* z, const Th* h, const Tm* m, LocalIndex numParticles, float G,
               Vec3<Tc> boxL, int numShells, Tc* ax, Tc* ay, Tc* az, Tc* ugrav)
{
#pragma omp parallel for schedule(static)
    for (LocalIndex t = 0; t < numParticles; ++t)
    {
        Vec4<Tc> acc{0, 0, 0, 0};
        Vec3<Tc> target{x[t], y[t], z[t]};
        Tc       target_h = h[t];

        for (int iz = -numShells; iz <= numShells; ++iz)
        {
            for (int iy = -numShells; iy <= numShells; ++iy)
            {
                for (int ix = -numShells; ix <= numShells; ++ix)
                {
                    auto targetRepl = target + Vec3<Tc>{ix * boxL[0], iy * boxL[1], iz * boxL[2]};

#if defined(__llvm__) || defined(__clang__)
#pragma clang loop vectorize(enable)
#endif
                    for (LocalIndex s = 0; s < numParticles; ++s)
                    {
                        acc = P2P(acc, targetRepl, Vec3<Tc>{x[s], y[s], z[s]}, m[s], target_h, h[s]);
                    }
                }
            }
        }

        *(ugrav + t) += G * m[t] * acc[0];
        *(ax + t) += G * acc[1];
        *(ay + t) += G * acc[2];
        *(az + t) += G * acc[3];
    }
}

} // namespace ryoanji
