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
#include "cstone/tree/octree_internal.hpp"
#include "cstone/focus/source_center.hpp"
#include "cartesian_qpole.hpp"

namespace ryoanji
{

/*! @brief computes gravitational acceleration for all particles in the specified group
 *
 * @tparam KeyType            unsigned 32- or 64-bit integer type
 * @tparam T1                 float or double
 * @tparam T2                 float or double
 * @param[in]    groupIdx     leaf cell index in [0:octree.numLeafNodes()] to compute accelerations for
 * @param[in]    octree       fully linked octree
 * @param[in]    multipoles   array of length @p octree.numTreeNodes() with the multipole moments for all nodes
 * @param[in]    layout       array of length @p octree.numLeafNodes()+1, layout[i] is the start offset
 *                            into the x,y,z,m arrays for the leaf node with index i. The last element
 *                            is equal to the length of the x,y,z,m arrays.
 * @param[in]    x            x-coordinates
 * @param[in]    y            y-coordinates
 * @param[in]    z            z-coordinates
 * @param[in]    h            smoothing lengths
 * @param[in]    m            masses
 * @param[in]    box          global coordinate bounding box
 * @param[in]    theta        accuracy parameter
 * @param[in]    G            gravitational constant
 * @param[inout] ax           location to add x-acceleration to
 * @param[inout] ay           location to add y-acceleration to
 * @param[inout] az           location to add z-acceleration to
 * @param[inout] ugrav        location to add gravitational potential to
 *
 * Note: acceleration output is added to destination
 */
template<class KeyType, class MType, class T1, class T2>
void computeGravityGroup(TreeNodeIndex groupIdx, const cstone::Octree<KeyType>& octree,
                         const cstone::SourceCenterType<T1>* centers, MType* multipoles, const LocalIndex* layout,
                         const T1* x, const T1* y, const T1* z, const T2* h, const T2* m, float G, T1* ax, T1* ay,
                         T1* az, T1* ugrav)
{
    LocalIndex firstTarget = layout[groupIdx];
    LocalIndex lastTarget  = layout[groupIdx + 1];

    Vec3<T1> tMin{x[firstTarget], y[firstTarget], z[firstTarget]};
    Vec3<T1> tMax = tMin;
    for (LocalIndex i = firstTarget; i < lastTarget; ++i)
    {
        Vec3<T1> tp{x[i], y[i], z[i]};
        tMin = min(tp, tMin);
        tMax = max(tp, tMax);
    }

    Vec3<T1> targetCenter = (tMax + tMin) * T2(0.5);
    Vec3<T1> targetSize   = (tMax - tMin) * T2(0.5);

    /*! @brief octree traversal continuation criterion
     *
     * This functor takes an octree node index as argument. It then evaluates the MAC between
     * the targetBox and the argument octree node. Returns true if the MAC failed to signal
     * the traversal routine to keep going. If the MAC passed, the multipole moments are applied
     * to the particles in the target box and traversal is stopped.
     */
    auto descendOrM2P =
        [firstTarget, lastTarget, centers, multipoles, x, y, z, G, ax, ay, az, ugrav, &targetCenter, &targetSize](
            TreeNodeIndex idx)
    {
        const auto& com = centers[idx];
        const auto& p   = multipoles[idx];

        bool violatesMac = cstone::evaluateMac(makeVec3(com), com[3], targetCenter, targetSize);

        if (!violatesMac)
        {
            LocalIndex numTargets = lastTarget - firstTarget;

// apply multipole to all particles in group
#if defined(__llvm__) || defined(__clang__)
#pragma clang loop vectorize(enable)
#endif
            for (LocalIndex t = 0; t < numTargets; ++t)
            {
                LocalIndex offset        = t + firstTarget;
                auto [ax_, ay_, az_, u_] = multipole2Particle(x[offset], y[offset], z[offset], makeVec3(com), p);
                *(ax + t) += G * ax_;
                *(ay + t) += G * ay_;
                *(az + t) += G * az_;
                *(ugrav + t) += G * u_;
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
    auto leafP2P = [groupIdx, layout, firstTarget, lastTarget, x, y, z, h, m, G, ax, ay, az, ugrav](TreeNodeIndex idx)
    {
        LocalIndex numTargets = lastTarget - firstTarget;

        LocalIndex firstSource = layout[idx];
        LocalIndex lastSource  = layout[idx + 1];
        LocalIndex numSources  = lastSource - firstSource;

        if (groupIdx != idx)
        {
            // source node != target node
            for (LocalIndex t = 0; t < numTargets; ++t)
            {
                LocalIndex offset        = t + firstTarget;
                auto [ax_, ay_, az_, u_] = particle2Particle(x[offset],
                                                             y[offset],
                                                             z[offset],
                                                             h[offset],
                                                             x + firstSource,
                                                             y + firstSource,
                                                             z + firstSource,
                                                             h + firstSource,
                                                             m + firstSource,
                                                             numSources);
                *(ax + t) += G * ax_;
                *(ay + t) += G * ay_;
                *(az + t) += G * az_;
                *(ugrav + t) += G * u_;
            }
        }
        else
        {
            assert(firstTarget == firstSource);
            // source node == target node -> source contains target, avoid self gravity
            for (LocalIndex t = 0; t < numTargets; ++t)
            {
                LocalIndex offset = t + firstTarget;
                // 2 splits: [firstSource:t] and [t+1:lastSource]
                auto [ax_, ay_, az_, u_] = particle2Particle(x[offset],
                                                             y[offset],
                                                             z[offset],
                                                             h[offset],
                                                             x + firstSource,
                                                             y + firstSource,
                                                             z + firstSource,
                                                             h + firstSource,
                                                             m + firstSource,
                                                             offset - firstSource);

                LocalIndex tp1               = offset + 1;
                auto [ax2_, ay2_, az2_, u2_] = particle2Particle(x[offset],
                                                                 y[offset],
                                                                 z[offset],
                                                                 h[offset],
                                                                 x + tp1,
                                                                 y + tp1,
                                                                 z + tp1,
                                                                 h + tp1,
                                                                 m + tp1,
                                                                 lastSource - tp1);
                *(ax + t) += G * (ax_ + ax2_);
                *(ay + t) += G * (ay_ + ay2_);
                *(az + t) += G * (az_ + az2_);
                *(ugrav + t) += G * (u_ + u2_);
            }
        }
    };

    singleTraversal(octree, descendOrM2P, leafP2P);
}

//! @brief repeats computeGravityGroup for all leaf node indices specified
template<class KeyType, class MType, class T1, class T2>
void computeGravity(const cstone::Octree<KeyType>& octree, const cstone::SourceCenterType<T1>* centers,
                    MType* multipoles, const LocalIndex* layout, TreeNodeIndex firstLeafIndex,
                    TreeNodeIndex lastLeafIndex, const T1* x, const T1* y, const T1* z, const T2* h, const T2* m,
                    float G, T1* ax, T1* ay, T1* az, T1* ugrav)
{
#pragma omp parallel for
    for (TreeNodeIndex leafIdx = firstLeafIndex; leafIdx < lastLeafIndex; ++leafIdx)
    {
        LocalIndex firstTarget = layout[leafIdx];
        computeGravityGroup(leafIdx,
                            octree,
                            centers,
                            multipoles,
                            layout,
                            x,
                            y,
                            z,
                            h,
                            m,
                            G,
                            ax + firstTarget,
                            ay + firstTarget,
                            az + firstTarget,
                            ugrav + firstTarget);
    }
}

/*! @brief repeats computeGravityGroup for all leaf node indices specified
 *
 * only computes total gravitational energy, no potential per particle
 *
 * @tparam KeyType               unsigned 32- or 64-bit integer type
 * @tparam T1                    float or double
 * @tparam T2                    float or double
 * @param[in]    octree          fully linked octree
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
 * @param[in]    G               gravitational constant
 * @param[inout] ax              location to add x-acceleration to
 * @param[inout] ay              location to add y-acceleration to
 * @param[inout] az              location to add z-acceleration to
 * @return                       total gravitational energy
 */
template<class KeyType, class MType, class T1, class T2>
T2 computeGravity(const cstone::Octree<KeyType>& octree, const cstone::SourceCenterType<T1>* centers, MType* multipoles,
                  const LocalIndex* layout, TreeNodeIndex firstLeafIndex, TreeNodeIndex lastLeafIndex, const T1* x,
                  const T1* y, const T1* z, const T2* h, const T2* m, float G, T1* ax, T1* ay, T1* az)
{
    T1 egravTot = 0.0;

    // determine maximum leaf particle count, bucketSize does not work, since octree might not be converged
    std::size_t maxNodeCount = 0;
#pragma omp parallel for reduction(max : maxNodeCount)
    for (TreeNodeIndex i = 0; i < octree.numLeafNodes(); ++i)
    {
        maxNodeCount = std::max(maxNodeCount, std::size_t(layout[i + 1] - layout[i]));
    }

#pragma omp parallel
    {
        T1 ugravThread[maxNodeCount];
        T1 egravThread = 0.0;

#pragma omp for
        for (TreeNodeIndex leafIdx = firstLeafIndex; leafIdx < lastLeafIndex; ++leafIdx)
        {
            LocalIndex firstTarget = layout[leafIdx];
            LocalIndex numTargets  = layout[leafIdx + 1] - firstTarget;

            std::fill(ugravThread, ugravThread + maxNodeCount, 0);
            computeGravityGroup(leafIdx,
                                octree,
                                centers,
                                multipoles,
                                layout,
                                x,
                                y,
                                z,
                                h,
                                m,
                                G,
                                ax + firstTarget,
                                ay + firstTarget,
                                az + firstTarget,
                                ugravThread);

            for (LocalIndex i = 0; i < numTargets; ++i)
            {
                egravThread += m[i + firstTarget] * ugravThread[i];
            }
        }

#pragma omp atomic
        egravTot += egravThread;
    }

    return 0.5 * egravTot;
}

//! @brief compute direct gravity sum for all particles [0:numParticles]
template<class T1, class T2>
void directSum(const T1* x, const T1* y, const T1* z, const T2* h, const T2* m, LocalIndex numParticles, float G,
               T1* ax, T1* ay, T1* az, T1* ugrav)
{
#pragma omp parallel for schedule(static)
    for (LocalIndex t = 0; t < numParticles; ++t)
    {
        // 2 splits: [0:t] and [t+1:numParticles]
        auto [ax_, ay_, az_, u_] = particle2Particle(x[t], y[t], z[t], h[t], x, y, z, h, m, t);

        LocalIndex tp1 = t + 1;
        auto [ax2_, ay2_, az2_, u2_] =
            particle2Particle(x[t], y[t], z[t], h[t], x + tp1, y + tp1, z + tp1, h + tp1, m + tp1, numParticles - tp1);

        *(ax + t) += G * (ax_ + ax2_);
        *(ay + t) += G * (ay_ + ay2_);
        *(az + t) += G * (az_ + az2_);
        *(ugrav + t) += G * m[t] * (u_ + u2_);
    }
}

} // namespace ryoanji
