/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Compute leaf cell source centers based on local information
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/util/array.hpp"
#include "cstone/tree/octree.hpp"
#include "cstone/traversal/macs.hpp"

namespace cstone
{

template<class T>
using SourceCenterType = util::array<T, 4>;

template<class Tc, class Th>
HOST_DEVICE_FUN util::tuple<Vec3<Tc>, Vec3<Tc>> computeBoundingBox(
    const Tc* x, const Tc* y, const Tc* z, const Th* h, LocalIndex first, LocalIndex last, Th scale, Vec3<Tc> init)
{
    if (first == last)
    {
        return {Vec3<Tc>{0, 0, 0}, Vec3<Tc>{0, 0, 0}};
    }

    Vec3<Tc> commonMin = init, commonMax = init;
    for (LocalIndex i = first; i < last; ++i)
    {
        auto r = h[i] * scale;
        Vec3<Tc> p{x[i], y[i], z[i]};
        commonMin = min(commonMin, Vec3<Tc>{p[0] - r, p[1] - r, p[2] - r});
        commonMax = max(commonMax, Vec3<Tc>{p[0] + r, p[1] + r, p[2] + r});
    }
    auto center = (commonMax + commonMin) * Tc(0.5);
    auto size   = (commonMax - commonMin) * Tc(0.5);
    return {center, size};
}

//! @brief add a single body contribution to a mass center
template<class T>
HOST_DEVICE_FUN void addBody(SourceCenterType<T>& center, const SourceCenterType<T>& source)
{
    T weight = std::abs(source[3]);

    center[0] += weight * source[0];
    center[1] += weight * source[1];
    center[2] += weight * source[2];
    center[3] += weight;
}

//! @brief finish mass center computation by dividing coordinates by total absolute mass
template<class T>
HOST_DEVICE_FUN SourceCenterType<T> normalizeMass(SourceCenterType<T> center)
{
    T invM = (center[3] != T(0.0)) ? T(1.0) / center[3] : T(1.0);
    center[0] *= invM;
    center[1] *= invM;
    center[2] *= invM;

    return center;
}

//! @brief compute a mass center from particles
template<class Ts, class Tc, class Tm>
HOST_DEVICE_FUN SourceCenterType<Ts>
massCenter(const Tc* x, const Tc* y, const Tc* z, const Tm* m, LocalIndex first, LocalIndex last)
{
    SourceCenterType<Ts> center{0, 0, 0, 0};
    for (LocalIndex i = first; i < last; ++i)
    {
        addBody(center, SourceCenterType<Ts>{Ts(x[i]), Ts(y[i]), Ts(z[i]), Ts(m[i])});
    }

    return normalizeMass(center);
}

//! @brief compute a mass center from other mass centers for use in tree upsweep
template<class T>
struct CombineSourceCenter
{
    HOST_DEVICE_FUN
    SourceCenterType<T> operator()(TreeNodeIndex /*nodeIdx*/, TreeNodeIndex child, const SourceCenterType<T>* centers)
    {
        SourceCenterType<T> center{0, 0, 0, 0};

        for (TreeNodeIndex i = child; i < child + 8; ++i)
        {
            addBody(center, centers[i]);
        }
        return normalizeMass(center);
    }
};

/*! @brief compute mass center coordinates for leaf nodes
 *
 * @param x                 source body x coordinates
 * @param y                 source body y coordinates
 * @param z                 source body z coordinates
 * @param m                 source body masses
 * @param leafToInternal    translation map from cornerstone leaf cell array indices to node indices of the full
 *                          octree
 * @param layout            array of length numLeafNodes + 1, the i-th element contains the index to of the first
 *                          particle in x,y,z,m contained in the i-th leaf node of the octree
 * @param sourceCenter      array of length numNodes of the full octree
 */
template<class T1, class T2, class T3>
void computeLeafMassCenter(std::span<const T1> x,
                           std::span<const T1> y,
                           std::span<const T1> z,
                           std::span<const T2> m,
                           std::span<const TreeNodeIndex> leafToInternal,
                           const LocalIndex* layout,
                           SourceCenterType<T3>* sourceCenter)
{
#pragma omp parallel for
    for (size_t leafIdx = 0; leafIdx < leafToInternal.size(); ++leafIdx)
    {
        TreeNodeIndex i = leafToInternal[leafIdx];
        sourceCenter[i] = massCenter<T3>(x.data(), y.data(), z.data(), m.data(), layout[leafIdx], layout[leafIdx + 1]);
    }
}

//! @brief replace the last center element (mass) with the squared mac radius
template<class T, class KeyType>
void setMac(std::span<const KeyType> nodeKeys,
            std::span<SourceCenterType<T>> centers,
            float invTheta,
            const Box<T>& box)
{
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nodeKeys.size(); ++i)
    {
        Vec4<T> center = centers[i];
        T mac          = computeVecMacR2(nodeKeys[i], util::makeVec3(center), invTheta, box);
        centers[i][3]  = (center[3] != T(0)) ? mac : T(0);
    }
}

//! @brief compute geometric node centers based on node SFC keys and the global bounding box
template<class KeyType, class T>
void nodeFpCenters(
    std::span<const KeyType> prefixes, Vec3<T>* centers, Vec3<T>* sizes, const Box<T>& box, bool disableMixD = false)
{
    const auto mixDBits = getBoxMixDimensionBits<T, KeyType, Box<T>>(box);
    const bool isMixD =
        !disableMixD && (mixDBits.bx != maxTreeLevel<KeyType>{} || mixDBits.by != maxTreeLevel<KeyType>{} ||
                         mixDBits.bz != maxTreeLevel<KeyType>{});
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < prefixes.size(); ++i)
    {
        KeyType prefix   = prefixes[i];
        KeyType startKey = decodePlaceholderBit(prefix);
        unsigned level   = decodePrefixLength(prefix) / 3;

        IBox nodeBox = isMixD ? sfcIBox(sfcMixDKey<KeyType>(startKey), maxTreeLevel<KeyType>{} - level, mixDBits.bx,
                                        mixDBits.by, mixDBits.bz)
                              : sfcIBox(sfcKey(startKey), level);
        if (nodeBox.xmin() == 0 && nodeBox.xmax() == 0 && nodeBox.ymin() == 0 && nodeBox.ymax() == 0 &&
            nodeBox.zmin() == 0 && nodeBox.zmax() == 0)
        {
            centers[i] = {0, 0, 0};
            sizes[i]   = {0, 0, 0};
        }
        else
        {
            util::tie(centers[i], sizes[i]) = centerAndSize<KeyType>(nodeBox, box, disableMixD);
        }
    }
}

template<class TreeType, class KeyType, class T>
std::pair<Vec3<T>, Vec3<T>> getCenterSizeMixDTree(TreeType tree, const TreeNodeIndex node, const Box<T>& box)
{
    KeyType startKey    = tree.codeStart(node);
    unsigned level      = tree.level(node);
    const auto mixDBits = getBoxMixDimensionBits<T, KeyType, Box<T>>(box);
    auto nodeBox =
        sfcIBox(sfcMixDKey<KeyType>(startKey), maxTreeLevel<KeyType>{} - level, mixDBits.bx, mixDBits.by, mixDBits.bz);
    auto [center, size] = centerAndSize<KeyType>(nodeBox, box, mixDBits.bx, mixDBits.by, mixDBits.bz);
    if (size[0] == 0 && size[1] == 0 && size[2] == 0) { tree.setEmpty(node); }
    return {center, size};
}

template<class KeyType, class T>
util::tuple<Vec3<T>, Vec3<T>> getCenterSizeMixD(const KeyType& prefix, const Box<T>& box)
{
    KeyType startKey            = decodePlaceholderBit(prefix);
    unsigned level              = decodePrefixLength(prefix) / 3;
    unsigned level_key          = octalDigit(startKey, level);
    const auto level_from_right = maxTreeLevel<KeyType>{} - level + 1;
    const auto mixDBits         = getBoxMixDimensionBits<T, KeyType, Box<T>>(box);
    unsigned sorted[3]          = {mixDBits.bx, mixDBits.by, mixDBits.bz};
    std::sort(std::begin(sorted), std::end(sorted));
    auto nodeBox = sfcIBox(sfcMixDKey<KeyType>(startKey), level_from_right - 1, mixDBits.bx, mixDBits.by, mixDBits.bz);
    Vec3<T> center, size;
    util::tie(center, size) = centerAndSize<KeyType>(nodeBox, box, mixDBits.bx, mixDBits.by, mixDBits.bz);
    if (level_from_right > sorted[2] && level_key > 0)
    {
        center = {0, 0, 0};
        size   = {0, 0, 0};
    }
    else if (level_from_right <= sorted[2] && level_from_right > sorted[1] && level_key > 1)
    {
        center = {0, 0, 0};
        size   = {0, 0, 0};
    }
    else if (level_from_right <= sorted[1] && level_from_right > sorted[0] && level_key > 3)
    {
        center = {0, 0, 0};
        size   = {0, 0, 0};
    }
    return {center, size};
}

//! @brief set @p centers to geometric node centers with Mac radius l * invTheta
template<class KeyType, class T>
void geoMacSpheres(std::span<const KeyType> prefixes, SourceCenterType<T>* centers, float invTheta, const Box<T>& box)
{
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < prefixes.size(); ++i)
    {
        centers[i] = computeMinMacR2(prefixes[i], invTheta, box);
    }
}

} // namespace cstone
