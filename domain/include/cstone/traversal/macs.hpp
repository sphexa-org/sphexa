/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Evaluate multipole acceptance criteria (MAC) on octree nodes
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#pragma once

#include "boxoverlap.hpp"
#include "cstone/primitives/math.hpp"
#include "cstone/traversal/traversal.hpp"
#include "cstone/tree/octree.hpp"

namespace cstone
{

//! @brief compute 1/theta + s for the minimum distance MAC
HOST_DEVICE_FUN inline float invThetaMinMac(float theta) { return 1.0f / theta + 0.5f; }

//! @brief cover worst-case vector mac with a min-Mac
HOST_DEVICE_FUN inline float invThetaMinToVec(float theta) { return 1.0f / theta + std::sqrt(3.0f) / 2; }

/*! @brief Compute square of the acceptance radius for the minimum distance MAC
 *
 * @param prefix       SFC key of the tree cell with Warren-Salmon placeholder-bit
 * @param invThetaEff  1/theta + s (effective opening parameter)
 * @param box          global coordinate bounding box
 * @return             geometric center in the first 3 elements, the square of the distance from @p sourceCenter
 *                     beyond which the MAC fails or passes in the 4th element
 */
template<class T, class KeyType>
HOST_DEVICE_FUN Vec4<T> computeMinMacR2(KeyType prefix, float invThetaEff, const Box<T>& box)
{
    KeyType nodeKey  = decodePlaceholderBit(prefix);
    int prefixLength = decodePrefixLength(prefix);

    IBox cellBox              = sfcIBox(sfcKey(nodeKey), prefixLength / 3);
    auto [geoCenter, geoSize] = centerAndSize<KeyType>(cellBox, box);

    T l   = T(2) * max(geoSize);
    T mac = l * invThetaEff;

    return {geoCenter[0], geoCenter[1], geoCenter[2], mac * mac};
}

/*! @brief Compute square of the acceptance radius for the vector MAC
 *
 * @param prefix       SFC key of the tree cell with Warren-Salmon placeholder-bit
 * @param expCenter    expansion (com) center of the source (cell)
 * @param invTheta     1/theta (opening parameter)
 * @param box          global coordinate bounding box
 * @return             the square of the distance from @p sourceCenter beyond which the MAC fails or passes
 */
template<class T, class KeyType>
HOST_DEVICE_FUN T computeVecMacR2(KeyType prefix, Vec3<T> expCenter, float invTheta, const Box<T>& box)
{
    KeyType nodeKey  = decodePlaceholderBit(prefix);
    int prefixLength = decodePrefixLength(prefix);

    IBox cellBox              = sfcIBox(sfcKey(nodeKey), prefixLength / 3);
    auto [geoCenter, geoSize] = centerAndSize<KeyType>(cellBox, box);

    Vec3<T> dX = expCenter - geoCenter;

    T s   = sqrt(norm2(dX));
    T l   = T(2.0) * max(geoSize);
    T mac = l * invTheta + s;

    return mac * mac;
}

/*! @brief evaluate an arbitrary MAC with respect to a given target
 *
 * @tparam T             float or double
 * @param sourceCenter   expansion center of the MAC
 * @param macSq          squared acceptance radius around @p sourceCenter
 * @param targetCenter   target coordinate
 * @param targetSize     target half box length (>0) in all dimensions
 * @return                true if the target is closer to @p sourceCenter than the acceptance radius
 */
template<class T>
HOST_DEVICE_FUN bool evaluateMac(Vec3<T> sourceCenter, T macSq, Vec3<T> targetCenter, Vec3<T> targetSize)
{
    Vec3<T> dX = abs(targetCenter - sourceCenter) - targetSize;
    dX += abs(dX);
    dX *= T(0.5);
    T R2 = norm2(dX);
    return R2 < std::abs(macSq);
}

/*! @brief evaluate an arbitrary MAC with respect to a given target
 *
 * @tparam T              float or double
 * @param  sourceCenter   source cell expansion center, can be geometric or center-mass, depending on
 *                        choice of MAC used to compute @p macSq
 * @param  macSq          squared multipole acceptance radius of the source cell
 * @param  targetCenter   geometric target cell center coordinates
 * @param  targetSize     geometric size of the target cell
 * @param  box            global coordinate bounding box
 * @return                true if the target is closer to @p sourceCenter than the acceptance radius
 */
template<class T>
HOST_DEVICE_FUN bool
evaluateMacPbc(Vec3<T> sourceCenter, T macSq, Vec3<T> targetCenter, Vec3<T> targetSize, const Box<T>& box)
{
    Vec3<T> dX = targetCenter - sourceCenter;

    dX = abs(applyPbc(dX, box));
    dX -= targetSize;
    dX += abs(dX);
    dX *= T(0.5);
    T R2 = norm2(dX);
    return R2 < std::abs(macSq);
}

/*! @brief integer based mutual min-mac
 * @tparam T float or double
 * @param a         first integer cell box
 * @param b         second integer cell box
 * @param ellipse   grid anisotropy (max grid-step in any dim / grid-step in dim_i) divided by theta
 *                  is equal to (L_max / L_x,y,z) * 1/theta if the number of grid points is equal in all dimensions
 * @param pbc       pbc yes/no per dimension
 * @return          true if MAC passed, i.e. true if cells are far
 */
template<class T>
HOST_DEVICE_FUN bool minMacMutualInt(IBox a, IBox b, Vec3<T> ellipse, Vec3<int> pbc)
{
    T l_max = std::max({a.xmax() - a.xmin(), a.ymax() - a.ymin(), a.zmax() - a.zmin(), b.xmax() - b.xmin(),
                        b.ymax() - b.ymin(), b.zmax() - b.zmin()});

    // computing a-b separation in integers is key to avoiding a-b/b-a asymmetry due to round-off errors
    Vec3<int> a_b = boxSeparation(a, b, pbc);

    Vec3<T> E{a_b[0] / ellipse[0], a_b[1] / ellipse[1], a_b[2] / ellipse[2]};
    return norm2(E) > l_max * l_max;
}

//! @brief mark all nodes of @p octree (leaves and internal) that fail the evaluateMac w.r.t to @p target
template<class T, class KeyType>
HOST_DEVICE_FUN void markMacPerBox(const Vec3<T>& targetCenter,
                                   const Vec3<T>& targetSize,
                                   unsigned maxSourceLevel,
                                   const KeyType* prefixes,
                                   const TreeNodeIndex* childOffsets,
                                   const TreeNodeIndex* parents,
                                   const Vec4<T>* centers,
                                   const Box<T>& box,
                                   KeyType focusStart,
                                   KeyType focusEnd,
                                   uint8_t* markings)
{
    auto checkAndMarkMac = [&](TreeNodeIndex idx)
    {
        KeyType nodePrefix   = prefixes[idx];
        unsigned sourceLevel = decodePrefixLength(nodePrefix) / 3;
        KeyType nodeStart    = decodePlaceholderBit(nodePrefix);
        KeyType nodeEnd      = nodeStart + nodeRange<KeyType>(sourceLevel);
        // if the tree node with index idx is fully contained in the focus, we stop traversal
        if (containedIn(nodeStart, nodeEnd, focusStart, focusEnd)) { return false; }

        Vec4<T> center = centers[idx];
        bool violatesMac =
            evaluateMacPbc(makeVec3(center), center[3], targetCenter, targetSize, box) && sourceLevel <= maxSourceLevel;
        if (violatesMac && !markings[idx]) { markings[idx] = 1; }

        return violatesMac;
    };

    singleTraversal(childOffsets, parents, checkAndMarkMac, [](TreeNodeIndex) {});
}

/*! @brief Mark each node in an octree that fails the MAC paired with any node from a given focus SFC range
 *
 * @tparam     T            float or double
 * @tparam     KeyType      32- or 64-bit unsigned integer
 * @param[in]  prefixes     SFC key for each tree cell with WS prefix bit
 * @param[in]  childOffsets index of first child for each node
 * @param[in]  parents      parent of each node i, stored at index (i-1)/8
 * @param[in]  centers      tree cell expansion (com) center coordinates and mac radius, size @p octree.numTreeNodes()
 * @param[in]  box          global coordinate bounding box
 * @param[in]  focusNodes   pointer to first LET leaf node in focus
 * @param[in]  numFocusNodes number of LET leaf nodes in focus
 * @param[in]  limitSource  if true, source cells are only marked if the tree-level is bigger than the target
 * @param[out] markings     array of length @p octree.numTreeNodes(), each position i
 *                          will be set to 1, if the node of @p octree with index i fails the MAC paired with
 *                          any node contained in the focus range [focusStart:focusEnd]
 */
template<class T, class KeyType>
void markMacs(const KeyType* prefixes,
              const TreeNodeIndex* childOffsets,
              const TreeNodeIndex* parents,
              const Vec4<T>* centers,
              const Box<T>& box,
              const KeyType* focusNodes,
              TreeNodeIndex numFocusNodes,
              bool limitSource,
              uint8_t* markings)
{
    KeyType focusStart = focusNodes[0];
    KeyType focusEnd   = focusNodes[numFocusNodes];

#pragma omp parallel for schedule(dynamic)
    for (TreeNodeIndex i = 0; i < numFocusNodes; ++i)
    {
        IBox target    = sfcIBox(sfcKey(focusNodes[i]), sfcKey(focusNodes[i + 1]));
        IBox targetExt = IBox(target.xmin() - 1, target.xmax() + 1, target.ymin() - 1, target.ymax() + 1,
                              target.zmin() - 1, target.zmax() + 1);
        if (containedIn(focusStart, focusEnd, targetExt)) { continue; }

        auto [targetCenter, targetSize] = centerAndSize<KeyType>(target, box);
        unsigned maxLevel               = maxTreeLevel<KeyType>{};
        if (limitSource) { maxLevel = std::max(int(treeLevel(focusNodes[i + 1] - focusNodes[i])) - 1, 0); }
        markMacPerBox(targetCenter, targetSize, maxLevel, prefixes, childOffsets, parents, centers, box, focusStart,
                      focusEnd, markings);
    }
}

} // namespace cstone
