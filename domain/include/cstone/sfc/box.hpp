/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief implements a bounding bounding box for floating point coordinates and integer indices
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cassert>
#include <cmath>

#include "cstone/cuda/annotation.hpp"
#include "cstone/primitives/stl.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/util/tuple.hpp"

namespace cstone
{

/*! @brief normalize a spatial length w.r.t. to a min/max range
 *
 * @tparam T
 * @param d
 * @param min
 * @param max
 * @return
 */
template<class T>
HOST_DEVICE_FUN constexpr T normalize(T d, T min, T max)
{
    return (d - min) / (max - min);
}

/*! @brief map x into periodic range 0...R-1
 *
 * @tparam R periodic range
 * @param x  input value
 * @return   x mapped into periodic range
 *
 * Examples:
 *   -1 -> R-1
 *    0 -> 0
 *    1 -> 1
 *  R-1 -> R-1
 *    R -> 0
 *  R+1 -> 1
 */
template<int R>
HOST_DEVICE_FUN constexpr int pbcAdjust(int x)
{
    // this version handles x outside -R, 2R
    // return x - R * std::floor(double(x)/R);
    assert(x >= -R);
    assert(x < 2 * R);
    int ret = (x < 0) ? x + R : x;
    return (ret >= R) ? ret - R : ret;
}

//! @brief maps x into the range [-R/2+1: R/2+1] (-511 to 512 with R = 1024)
HOST_DEVICE_FUN constexpr int pbcDistance(int x, int R)
{
    // this version handles x outside -R, R
    // int roundAwayFromZero = (x > 0) ? x + R/2 : x - R/2;
    // return x -= R * (roundAwayFromZero / R);
    assert(R == 0 || x >= -R);
    assert(R == 0 || x <= R);
    int ret = (x <= -R / 2) ? x + R : x;
    return (ret > R / 2) ? ret - R : ret;
}

enum class BoundaryType : char
{
    open     = 0,
    periodic = 1,
    fixed    = 2
};

/*! @brief stores the coordinate bounds
 *
 * Needs a slightly different behavior in the PBC case than the existing BBox
 * to manage morton code based octrees.
 *
 * @tparam T floating point type
 */
template<class T>
class Box
{

public:
    HOST_DEVICE_FUN constexpr Box(T xyzMin, T xyzMax, BoundaryType b = BoundaryType::open)
        : limits{xyzMin, xyzMax, xyzMin, xyzMax, xyzMin, xyzMax}
        , lengths_{xyzMax - xyzMin, xyzMax - xyzMin, xyzMax - xyzMin}
        , inverseLengths_{T(1.) / (xyzMax - xyzMin), T(1.) / (xyzMax - xyzMin), T(1.) / (xyzMax - xyzMin)}
        , boundaries{b, b, b}
    {
    }

    HOST_DEVICE_FUN constexpr Box(T xmin,
                                  T xmax,
                                  T ymin,
                                  T ymax,
                                  T zmin,
                                  T zmax,
                                  BoundaryType bx = BoundaryType::open,
                                  BoundaryType by = BoundaryType::open,
                                  BoundaryType bz = BoundaryType::open)
        : limits{xmin, xmax, ymin, ymax, zmin, zmax}
        , lengths_{xmax - xmin, ymax - ymin, zmax - zmin}
        , inverseLengths_{T(1.) / (xmax - xmin), T(1.) / (ymax - ymin), T(1.) / (zmax - zmin)}
        , boundaries{bx, by, bz}
    {
    }

    HOST_DEVICE_FUN constexpr T xmin() const { return limits[0]; }
    HOST_DEVICE_FUN constexpr T xmax() const { return limits[1]; }
    HOST_DEVICE_FUN constexpr T ymin() const { return limits[2]; }
    HOST_DEVICE_FUN constexpr T ymax() const { return limits[3]; }
    HOST_DEVICE_FUN constexpr T zmin() const { return limits[4]; }
    HOST_DEVICE_FUN constexpr T zmax() const { return limits[5]; }

    //! @brief return edge lengths
    HOST_DEVICE_FUN constexpr T lx() const { return lengths_[0]; }
    HOST_DEVICE_FUN constexpr T ly() const { return lengths_[1]; }
    HOST_DEVICE_FUN constexpr T lz() const { return lengths_[2]; }

    //! @brief return inverse edge lengths
    HOST_DEVICE_FUN constexpr T ilx() const { return inverseLengths_[0]; }
    HOST_DEVICE_FUN constexpr T ily() const { return inverseLengths_[1]; }
    HOST_DEVICE_FUN constexpr T ilz() const { return inverseLengths_[2]; }

    HOST_DEVICE_FUN constexpr BoundaryType boundaryX() const { return boundaries[0]; } // NOLINT
    HOST_DEVICE_FUN constexpr BoundaryType boundaryY() const { return boundaries[1]; } // NOLINT
    HOST_DEVICE_FUN constexpr BoundaryType boundaryZ() const { return boundaries[2]; } // NOLINT

    //! @brief return the shortest coordinate range in any dimension
    HOST_DEVICE_FUN constexpr T minExtent() const { return stl::min(stl::min(lengths_[0], lengths_[1]), lengths_[2]); }

    //! @brief return the longes coordinate range in any dimension
    HOST_DEVICE_FUN constexpr T maxExtent() const { return stl::max(stl::max(lengths_[0], lengths_[1]), lengths_[2]); }

    template<class Archive>
    void loadOrStore(Archive* ar)
    {
        ar->stepAttribute("box", limits, 6);
        ar->stepAttribute("boundaryType", (char*)boundaries, 3);

        *this = Box<T>(limits[0], limits[1], limits[2], limits[3], limits[4], limits[5], boundaries[0], boundaries[1],
                       boundaries[2]);
    }

private:
    HOST_DEVICE_FUN
    friend constexpr bool operator==(const Box<T>& a, const Box<T>& b)
    {
        return a.limits[0] == b.limits[0] && a.limits[1] == b.limits[1] && a.limits[2] == b.limits[2] &&
               a.limits[3] == b.limits[3] && a.limits[4] == b.limits[4] && a.limits[5] == b.limits[5] &&
               a.boundaries[0] == b.boundaries[0] && a.boundaries[1] == b.boundaries[1] &&
               a.boundaries[2] == b.boundaries[2];
    }

    T limits[6];
    T lengths_[3];
    T inverseLengths_[3];
    BoundaryType boundaries[3];
};

//! @brief Compute the shortest periodic distance dX = A - B between two points,
template<class T>
HOST_DEVICE_FUN inline Vec3<T> applyPbc(Vec3<T> dX, const Box<T>& box)
{
    bool pbcX = (box.boundaryX() == BoundaryType::periodic);
    bool pbcY = (box.boundaryY() == BoundaryType::periodic);
    bool pbcZ = (box.boundaryZ() == BoundaryType::periodic);

    dX[0] -= pbcX * box.lx() * std::rint(dX[0] * box.ilx());
    dX[1] -= pbcY * box.ly() * std::rint(dX[1] * box.ily());
    dX[2] -= pbcZ * box.lz() * std::rint(dX[2] * box.ilz());

    return dX;
}

//! @brief Fold X into a periodic image that lies inside @a box
template<class T>
HOST_DEVICE_FUN inline Vec3<T> putInBox(Vec3<T> X, const Box<T>& box)
{
    bool pbcX = (box.boundaryX() == BoundaryType::periodic);
    bool pbcY = (box.boundaryY() == BoundaryType::periodic);
    bool pbcZ = (box.boundaryZ() == BoundaryType::periodic);

    // Further testing needed before this can be enabled
    // X[0] -= pbcX * box.lx() * std::trunc(X[0] * box.ilx());
    // X[1] -= pbcY * box.ly() * std::trunc(X[1] * box.ily());
    // X[2] -= pbcZ * box.lz() * std::trunc(X[2] * box.ilz());

    if (pbcX && X[0] > box.xmax()) { X[0] -= box.lx(); }
    else if (pbcX && X[0] < box.xmin()) { X[0] += box.lx(); }

    if (pbcY && X[1] > box.ymax()) { X[1] -= box.ly(); }
    else if (pbcY && X[1] < box.ymin()) { X[1] += box.ly(); }

    if (pbcZ && X[2] > box.zmax()) { X[2] -= box.lz(); }
    else if (pbcZ && X[2] < box.zmin()) { X[2] += box.lz(); }

    return X;
}

//! @brief Legacy PBC
template<class Tc, class T>
HOST_DEVICE_FUN inline void applyPBC(const cstone::Box<Tc>& box, T r, T& xx, T& yy, T& zz)
{
    bool pbcX = (box.boundaryX() == BoundaryType::periodic);
    bool pbcY = (box.boundaryY() == BoundaryType::periodic);
    bool pbcZ = (box.boundaryZ() == BoundaryType::periodic);

    if (pbcX && xx > r)
        xx -= box.lx();
    else if (pbcX && xx < -r)
        xx += box.lx();

    if (pbcY && yy > r)
        yy -= box.ly();
    else if (pbcY && yy < -r)
        yy += box.ly();

    if (pbcZ && zz > r)
        zz -= box.lz();
    else if (pbcZ && zz < -r)
        zz += box.lz();
}

template<class Tc, class T>
HOST_DEVICE_FUN inline T distancePBC(const cstone::Box<Tc>& box, T hi, Tc x1, Tc y1, Tc z1, Tc x2, Tc y2, Tc z2)
{
    T xx = x1 - x2;
    T yy = y1 - y2;
    T zz = z1 - z2;

    applyPBC(box, T(2) * hi, xx, yy, zz);

    return std::sqrt(xx * xx + yy * yy + zz * zz);
}

/*! @brief stores octree index integer bounds
 */
template<class T>
class SimpleBox
{
public:
    HOST_DEVICE_FUN constexpr SimpleBox()
        : limits{0, 0, 0, 0, 0, 0}
    {
    }

    HOST_DEVICE_FUN constexpr SimpleBox(T xyzMin, T xyzMax)
        : limits{xyzMin, xyzMax, xyzMin, xyzMax, xyzMin, xyzMax}
    {
    }

    HOST_DEVICE_FUN constexpr SimpleBox(T xmin, T xmax, T ymin, T ymax, T zmin, T zmax)
        : limits{xmin, xmax, ymin, ymax, zmin, zmax}
    {
    }

    HOST_DEVICE_FUN constexpr T xmin() const { return limits[0]; } // NOLINT
    HOST_DEVICE_FUN constexpr T xmax() const { return limits[1]; } // NOLINT
    HOST_DEVICE_FUN constexpr T ymin() const { return limits[2]; } // NOLINT
    HOST_DEVICE_FUN constexpr T ymax() const { return limits[3]; } // NOLINT
    HOST_DEVICE_FUN constexpr T zmin() const { return limits[4]; } // NOLINT
    HOST_DEVICE_FUN constexpr T zmax() const { return limits[5]; } // NOLINT

    //! @brief return the shortest coordinate range in any dimension
    HOST_DEVICE_FUN constexpr T minExtent() const // NOLINT
    {
        return stl::min(stl::min(xmax() - xmin(), ymax() - ymin()), zmax() - zmin());
    }

private:
    HOST_DEVICE_FUN
    friend constexpr bool operator==(const SimpleBox& a, const SimpleBox& b)
    {
        return a.limits[0] == b.limits[0] && a.limits[1] == b.limits[1] && a.limits[2] == b.limits[2] &&
               a.limits[3] == b.limits[3] && a.limits[4] == b.limits[4] && a.limits[5] == b.limits[5];
    }

    HOST_DEVICE_FUN
    friend bool operator<(const SimpleBox& a, const SimpleBox& b)
    {
        return util::tie(a.limits[0], a.limits[1], a.limits[2], a.limits[3], a.limits[4], a.limits[5]) <
               util::tie(b.limits[0], b.limits[1], b.limits[2], b.limits[3], b.limits[4], b.limits[5]);
    }

    T limits[6];
};

using IBox = SimpleBox<int>;

template<class T>
using FBox = SimpleBox<T>;

/*! @brief calculate floating point 3D center and radius of a and integer box and bounding box pair
 *
 * @tparam T         float or double
 * @tparam KeyType   32- or 64-bit unsigned integer
 * @param ibox       integer coordinate box
 * @param box        floating point bounding box
 * @return           the geometrical center and the vector from the center to the box corner farthest from the origin
 */
template<class KeyType, class T>
constexpr HOST_DEVICE_FUN util::tuple<Vec3<T>, Vec3<T>> centerAndSize(const IBox& ibox, const Box<T>& box)
{
    constexpr int maxCoord = 1u << maxTreeLevel<KeyType>{};
    // smallest octree cell edge length in unit cube
    constexpr T uL = T(1.) / maxCoord;

    T halfUnitLengthX = T(0.5) * uL * box.lx();
    T halfUnitLengthY = T(0.5) * uL * box.ly();
    T halfUnitLengthZ = T(0.5) * uL * box.lz();
    Vec3<T> boxCenter = {box.xmin() + (ibox.xmax() + ibox.xmin()) * halfUnitLengthX,
                         box.ymin() + (ibox.ymax() + ibox.ymin()) * halfUnitLengthY,
                         box.zmin() + (ibox.zmax() + ibox.zmin()) * halfUnitLengthZ};
    Vec3<T> boxSize   = {(ibox.xmax() - ibox.xmin()) * halfUnitLengthX, (ibox.ymax() - ibox.ymin()) * halfUnitLengthY,
                         (ibox.zmax() - ibox.zmin()) * halfUnitLengthZ};

    return {boxCenter, boxSize};
}

/*! @brief create a floating point box from and integer box
 *
 * @tparam T         float or double
 * @tparam KeyType   32- or 64-bit unsigned integer
 * @param ibox       integer box
 * @param box        global coordinate bounding box
 * @return           the floating point box
 */
template<class KeyType, class T>
constexpr HOST_DEVICE_FUN FBox<T> createFpBox(const IBox& ibox, const Box<T>& box)
{
    auto [center, size] = centerAndSize<KeyType>(ibox, box);

    auto Xmin = center - size;
    auto Xmax = center + size;

    return {Xmin[0], Xmax[0], Xmin[1], Xmax[1], Xmin[2], Xmax[2]};
}

/*! @brief convert a floating point box to an IBox with a volume not smaller than the input box
 *
 * @param center  floating point box center
 * @param size    floating point box size
 * @param box     global coordinate bounding box
 * @return        the converted IBox
 *
 * Inverts createFpBox
 */
template<class KeyType, class T>
HOST_DEVICE_FUN IBox createIBox(const Vec3<T> center, const Vec3<T>& size, const Box<T>& box)
{
    constexpr int maxCoord = 1u << maxTreeLevel<KeyType>{};

    Vec3<T> Xmin = center - size;
    Vec3<T> Xmax = center + size;

    // normalize to units of box lengths
    T xnMin = (Xmin[0] - box.xmin()) * box.ilx();
    T ynMin = (Xmin[1] - box.ymin()) * box.ily();
    T znMin = (Xmin[2] - box.zmin()) * box.ilz();

    T xnMax = (Xmax[0] - box.xmin()) * box.ilx();
    T ynMax = (Xmax[1] - box.ymin()) * box.ily();
    T znMax = (Xmax[2] - box.zmin()) * box.ilz();

    int ixMin = std::floor(xnMin * maxCoord);
    int iyMin = std::floor(ynMin * maxCoord);
    int izMin = std::floor(znMin * maxCoord);

    int ixMax = std::ceil(xnMax * maxCoord);
    int iyMax = std::ceil(ynMax * maxCoord);
    int izMax = std::ceil(znMax * maxCoord);

    return {ixMin, ixMax, iyMin, iyMax, izMin, izMax};
}

/*! @brief limit shrinking of the box due to removal of particles
 * @param fittingBox tight box for the current system
 * @param previousBox box used for the previous time-step
 * @param shrinkLimit maximum shrink factor per side compared to previousBox
 */
template<typename T>
Box<T> limitBoxShrinking(const Box<T>& fittingBox, const Box<T>& previousBox, const T shrinkLimit = 0.05)
{
    const std::array<T, 6> limits{
        previousBox.xmin() + shrinkLimit * previousBox.lx(), previousBox.xmax() - shrinkLimit * previousBox.lx(),
        previousBox.ymin() + shrinkLimit * previousBox.ly(), previousBox.ymax() - shrinkLimit * previousBox.ly(),
        previousBox.zmin() + shrinkLimit * previousBox.lz(), previousBox.zmax() - shrinkLimit * previousBox.lz()};

    return Box<T>{std::min(fittingBox.xmin(), limits[0]),
                  std::max(fittingBox.xmax(), limits[1]),
                  std::min(fittingBox.ymin(), limits[2]),
                  std::max(fittingBox.ymax(), limits[3]),
                  std::min(fittingBox.zmin(), limits[4]),
                  std::max(fittingBox.zmax(), limits[5]),
                  previousBox.boundaryX(),
                  previousBox.boundaryY(),
                  previousBox.boundaryZ()};
}

} // namespace cstone
