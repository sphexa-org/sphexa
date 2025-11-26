/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief (halo-)box overlap functionality
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/sfc/sfc.hpp"

namespace cstone
{

//! @brief standard criterion for two ranges a-b and c-d to overlap, a<b and c<d
template<class T>
HOST_DEVICE_FUN constexpr bool overlapTwoRanges(T a, T b, T c, T d)
{
    assert(a <= b && c <= d);
    return b > c && d > a;
}

/*! @brief determine whether two ranges ab and cd overlap
 *
 * @param R  periodic range
 * @return   true if ranges overlap, false otherwise
 *
 * Some restrictions apply, no input value can be further than R from the periodic range.
 */
HOST_DEVICE_FUN constexpr bool overlapRange(int a, int b, int c, int d, int R)
{
    assert(a >= -R);
    assert(a < R);
    assert(b > 0);
    assert(b <= 2 * R);

    assert(c >= -R);
    assert(c < R);
    assert(d > 0);
    assert(d <= 2 * R);

    return overlapTwoRanges(a, b, c, d) || overlapTwoRanges(a + R, b + R, c, d) || overlapTwoRanges(a, b, c + R, d + R);
}

//! @brief compute minimal separation between ranges [a,b] and [c,d], R=0 means no periodicity
HOST_DEVICE_FUN inline int rangeSep(int a, int b, int c, int d, int R)
{
    assert(a <= b && c <= d);
    if (overlapTwoRanges(a, b, c, d)) { return 0; }
    int d1 = pbcDistance(d - a, R);
    int d2 = pbcDistance(c - b, R);
    return stl::min(std::abs(d1), std::abs(d2));
}

//! @brief check whether two boxes overlap. takes PBC into account, boxes can wrap around
template<class KeyType>
HOST_DEVICE_FUN bool overlap(const IBox& a, const IBox& b)
{
    constexpr int maxCoord = 1u << maxTreeLevel<KeyType>{};

    bool xOverlap = overlapRange(a.xmin(), a.xmax(), b.xmin(), b.xmax(), maxCoord);
    bool yOverlap = overlapRange(a.ymin(), a.ymax(), b.ymin(), b.ymax(), maxCoord);
    bool zOverlap = overlapRange(a.zmin(), a.zmax(), b.zmin(), b.zmax(), maxCoord);

    return xOverlap && yOverlap && zOverlap;
}

//! @brief return separation between integer boxes a, b. @p pbc is the grid periodicity in each dimension
HOST_DEVICE_FUN inline Vec3<int> boxSeparation(IBox a, IBox b, Vec3<int> pbc)
{
    int dx = rangeSep(a.xmin(), a.xmax(), b.xmin(), b.xmax(), pbc[0]);
    int dy = rangeSep(a.ymin(), a.ymax(), b.ymin(), b.ymax(), pbc[1]);
    int dz = rangeSep(a.zmin(), a.zmax(), b.zmin(), b.zmax(), pbc[2]);
    return {dx, dy, dz};
}

/*! @brief Check whether a coordinate box is fully contained in a Morton code range
 *
 * @tparam KeyType   32- or 64-bit unsigned integer
 * @param codeStart  Morton code range start
 * @param codeEnd    Morton code range end
 * @param box        3D box with x,y,z integer coordinates in [0,2^maxTreeLevel<KeyType>{}-1]
 * @return           true if the box is fully contained within the specified Morton code range
 */
template<class KeyType>
HOST_DEVICE_FUN std::enable_if_t<std::is_unsigned_v<KeyType>, bool>
containedIn(KeyType codeStart, KeyType codeEnd, const IBox& box)
{
    // volume 0 boxes are not possible if makeHaloBox was used to generate it
    assert(box.xmin() < box.xmax());
    assert(box.ymin() < box.ymax());
    assert(box.zmin() < box.zmax());

    constexpr int pbcRange = 1 << maxTreeLevel<KeyType>{};
    if (stl::min(stl::min(box.xmin(), box.ymin()), box.zmin()) < 0 ||
        stl::max(stl::max(box.xmax(), box.ymax()), box.zmax()) > pbcRange)
    {
        // any box that wraps around a PBC boundary cannot be contained within
        // any octree node, except the full root node
        return codeStart == 0 && codeEnd == nodeRange<KeyType>(0);
    }

    KeyType lowCode  = iSfcKey<SfcKind<KeyType>>(box.xmin(), box.ymin(), box.zmin());
    KeyType highCode = iSfcKey<SfcKind<KeyType>>(box.xmax() - 1, box.ymax() - 1, box.zmax() - 1);
    auto envelope    = smallestCommonBox(lowCode, highCode);

    return (util::get<0>(envelope) >= codeStart) && (util::get<1>(envelope) <= codeEnd);
}

/*! @brief Check whether a coordinate box is fully contained in a SFC key range
 *
 * @tparam KeyType   32- or 64-bit unsigned integer
 * @param codeStart  Morton code range start
 * @param codeEnd    Morton code range end
 * @param box        3D box with x,y,z integer coordinates in [0,2^maxTreeLevel<KeyType>{}-1]
 * @return           true if the box is fully contained within the specified Morton code range
 */
template<class KeyType, class Tc>
HOST_DEVICE_FUN std::enable_if_t<std::is_unsigned_v<KeyType>, bool>
containedIn(KeyType codeStart, KeyType codeEnd, const Vec3<Tc>& center, const Vec3<Tc>& size, const Box<Tc>& box)
{
    auto boxMin   = center - size;
    auto boxMax   = center + size;
    auto dFromMin = min(boxMin - Vec3<Tc>{box.xmin(), box.ymin(), box.zmin()});
    auto dFromMax = max(boxMax - Vec3<Tc>{box.xmax(), box.ymax(), box.zmax()});

    if (dFromMin < Tc(0) || dFromMax > Tc(0))
    {
        // any box that wraps around a PBC boundary cannot be contained within
        // any octree node, except the full root node
        return codeStart == 0 && codeEnd == nodeRange<KeyType>(0);
    }

    // increase maximum by a grid-unit to ensure we round up
    constexpr int gridDim = 1u << maxTreeLevel<KeyType>{};
    boxMax += Vec3<Tc>{box.lx(), box.ly(), box.lz()} * (Tc(1) / gridDim);

    KeyType lowCode  = sfc3D<SfcKind<KeyType>>(boxMin[0], boxMin[1], boxMin[2], box);
    KeyType highCode = sfc3D<SfcKind<KeyType>>(boxMax[0], boxMax[1], boxMax[2], box);
    auto envelope    = smallestCommonBox(lowCode, highCode);

    return (util::get<0>(envelope) >= codeStart) && (util::get<1>(envelope) <= codeEnd);
}

/*! @brief determine whether a binary/octree node (prefix, prefixLength) is fully contained in an SFC range
 *
 * @tparam KeyType       32- or 64-bit unsigned integer
 * @param  nodeStart     lowest SFC code of the tree node
 * @param  nodeEnd       highest SFC key of the tree node,
 * @param  codeStart     start of the SFC range
 * @param  codeEnd       end of the SFC range
 * @return
 */
template<class KeyType>
HOST_DEVICE_FUN std::enable_if_t<std::is_unsigned_v<KeyType>, bool>
containedIn(KeyType nodeStart, KeyType nodeEnd, KeyType codeStart, KeyType codeEnd)
{
    return !(nodeStart < codeStart || nodeEnd > codeEnd);
}

template<class KeyType>
HOST_DEVICE_FUN std::enable_if_t<std::is_unsigned_v<KeyType>, bool>
containedIn(KeyType prefixBitKey, KeyType codeStart, KeyType codeEnd)
{
    unsigned prefixLength = decodePrefixLength(prefixBitKey);
    KeyType firstPrefix   = decodePlaceholderBit(prefixBitKey);
    KeyType secondPrefix  = firstPrefix + (KeyType(1) << (3 * maxTreeLevel<KeyType>{} - prefixLength));
    return !(firstPrefix < codeStart || secondPrefix > codeEnd);
}

template<class KeyType>
HOST_DEVICE_FUN int addDelta(int value, int delta, bool pbc)
{
    constexpr int maxCoordinate = (1u << maxTreeLevel<KeyType>{});

    int temp = value + delta;
    if (pbc) { return temp; }
    else { return stl::min(stl::max(0, temp), maxCoordinate); }
}

//! @brief create a box with specified radius around node delineated by codeStart/End
template<class KeyType, class CoordinateType, class RadiusType>
HOST_DEVICE_FUN IBox makeHaloBox(const IBox& nodeBox, RadiusType radius, const Box<CoordinateType>& box)
{
    int dx = toNBitIntCeil<KeyType>(radius * box.ilx());
    int dy = toNBitIntCeil<KeyType>(radius * box.ily());
    int dz = toNBitIntCeil<KeyType>(radius * box.ilz());

    bool pbcX = (box.boundaryX() == cstone::BoundaryType::periodic);
    bool pbcY = (box.boundaryY() == cstone::BoundaryType::periodic);
    bool pbcZ = (box.boundaryZ() == cstone::BoundaryType::periodic);

    return IBox(addDelta<KeyType>(nodeBox.xmin(), -dx, pbcX), addDelta<KeyType>(nodeBox.xmax(), dx, pbcX),
                addDelta<KeyType>(nodeBox.ymin(), -dy, pbcY), addDelta<KeyType>(nodeBox.ymax(), dy, pbcY),
                addDelta<KeyType>(nodeBox.zmin(), -dz, pbcZ), addDelta<KeyType>(nodeBox.zmax(), dz, pbcZ));
}

//! @brief create a box with specified radius around node delineated by codeStart/End
template<class KeyType, class CoordinateType, class RadiusType>
HOST_DEVICE_FUN IBox makeHaloBox(KeyType codeStart, KeyType codeEnd, RadiusType radius, const Box<CoordinateType>& box)
{
    // disallow boxes with no volume
    assert(codeEnd > codeStart);
    IBox nodeBox = sfcIBox(sfcKey(codeStart), sfcKey(codeEnd));
    return makeHaloBox<KeyType>(nodeBox, radius, box);
}

//! @brief returns true if the cuboid defined by center and size is contained within the bounding box
template<class T>
HOST_DEVICE_FUN bool insideBox(const Vec3<T>& center, const Vec3<T>& size, const Box<T>& box)
{
    Vec3<T> globalMin{box.xmin(), box.ymin(), box.zmin()};
    Vec3<T> globalMax{box.xmax(), box.ymax(), box.zmax()};
    Vec3<T> boxMin = center - size;
    Vec3<T> boxMax = center + size;
    return boxMin[0] >= globalMin[0] && boxMin[1] >= globalMin[1] && boxMin[2] >= globalMin[2] &&
           boxMax[0] <= globalMax[0] && boxMax[1] <= globalMax[1] && boxMax[2] <= globalMax[2];
}

//! @brief returns the smallest distance vector of point X to box b, 0 if X is in b
template<class T>
HOST_DEVICE_FUN Vec3<T> minDistance(const Vec3<T>& X, const Vec3<T>& bCenter, const Vec3<T>& bSize)
{
    Vec3<T> dX = abs(bCenter - X) - bSize;
    dX += abs(dX);
    dX *= T(0.5);
    return dX;
}

//! @brief returns the smallest periodic distance vector of point X to box b, 0 if X is in b
template<class T>
HOST_DEVICE_FUN Vec3<T> minDistance(const Vec3<T>& X, const Vec3<T>& bCenter, const Vec3<T>& bSize, const Box<T>& box)
{
    Vec3<T> dX = bCenter - X;
    dX         = abs(applyPbc(dX, box));
    dX -= bSize;
    dX += abs(dX);
    dX *= T(0.5);

    return dX;
}

//! @brief returns the smallest distance vector between two boxes, 0 if they overlap
template<class T>
HOST_DEVICE_FUN Vec3<T>
minDistance(const Vec3<T>& aCenter, const Vec3<T>& aSize, const Vec3<T>& bCenter, const Vec3<T>& bSize)
{
    Vec3<T> dX = abs(bCenter - aCenter) - aSize - bSize;
    dX += abs(dX);
    dX *= T(0.5);

    return dX;
}

//! @brief returns the smallest periodic distance vector between two boxes, 0 if they overlap
template<class T>
HOST_DEVICE_FUN Vec3<T> minDistance(
    const Vec3<T>& aCenter, const Vec3<T>& aSize, const Vec3<T>& bCenter, const Vec3<T>& bSize, const Box<T>& box)
{
    Vec3<T> dX = bCenter - aCenter;
    dX         = abs(applyPbc(dX, box));
    dX -= aSize;
    dX -= bSize;
    dX += abs(dX);
    dX *= T(0.5);

    return dX;
}

//! @brief returns true if the two cuboids a and b overlap
template<class T>
HOST_DEVICE_FUN bool
overlap(const Vec3<T>& aCenter, const Vec3<T>& aSize, const Vec3<T>& bCenter, const Vec3<T>& bSize, const Box<T>& box)
{
    Vec3<T> dX = bCenter - aCenter;
    dX         = abs(applyPbc(dX, box));
    dX -= aSize;
    dX -= bSize;

    constexpr T eps = 0;
    return dX[0] < eps && dX[1] < eps && dX[2] < eps;
}

//! @brief Convenience wrapper to minDistance. This should only be used for testing.
template<class KeyType, class T>
HOST_DEVICE_FUN T minDistanceSq(IBox a, IBox b, const Box<T>& box)
{
    auto [aCenter, aSize] = centerAndSize<KeyType>(a, box);
    auto [bCenter, bSize] = centerAndSize<KeyType>(b, box);
    return norm2(minDistance(aCenter, aSize, bCenter, bSize, box));
}

} // namespace cstone
