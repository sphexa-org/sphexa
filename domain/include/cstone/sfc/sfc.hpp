/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  SFC encoding/decoding in 32- and 64-bit
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * Common interface to Morton and Hilbert keys based on strong C++ types
 */

#pragma once

#include "cstone/util/strong_type.hpp"

#include "box.hpp"
#include "morton.hpp"
#include "hilbert.hpp"

namespace cstone
{

//! @brief Strong type for Morton keys
template<class IntegerType>
using MortonKey = StrongType<IntegerType, struct MortonKeyTag>;

//! @brief Strong type for Hilbert keys
template<class IntegerType>
using HilbertKey = StrongType<IntegerType, struct HilbertKeyTag>;

//! @brief Strong type for Hilbert Mixed Dimension keys
template<class IntegerType>
using HilbertMixDKey = StrongType<IntegerType, struct HilbertMixDKeyTag>;

//! @brief use this definition to select the kind of space filling curve to use
template<class IntegerType>
using SfcKind = HilbertKey<IntegerType>;

template<class KeyType>
HOST_DEVICE_FUN SfcKind<KeyType> sfcKey(KeyType key)
{
    return SfcKind<KeyType>(key);
}

//! @brief convert an integer pointer to the corresponding strongly typed SFC key pointer
template<class KeyType>
HOST_DEVICE_FUN SfcKind<KeyType>* sfcKindPointer(KeyType* ptr)
{
    return reinterpret_cast<SfcKind<KeyType>*>(ptr);
}

//! @brief convert a integer pointer to the corresponding strongly typed SFC key pointer
template<class KeyType>
HOST_DEVICE_FUN const SfcKind<KeyType>* sfcKindPointer(const KeyType* ptr)
{
    return reinterpret_cast<const SfcKind<KeyType>*>(ptr);
}

template<class IntegerType>
using SfcMixDKind = HilbertMixDKey<IntegerType>;

template<class KeyType>
HOST_DEVICE_FUN SfcMixDKind<KeyType> sfcMixDKey(KeyType key)
{
    return SfcMixDKind<KeyType>(key);
}

//! @brief convert an integer pointer to the corresponding strongly typed SFC key pointer
template<class KeyType>
HOST_DEVICE_FUN SfcMixDKind<KeyType>* SfcMixDKindPointer(KeyType* ptr)
{
    return reinterpret_cast<SfcMixDKind<KeyType>*>(ptr);
}

//! @brief convert a integer pointer to the corresponding strongly typed SFC key pointer
template<class KeyType>
HOST_DEVICE_FUN const SfcMixDKind<KeyType>* SfcMixDKindPointer(const KeyType* ptr)
{
    return reinterpret_cast<const SfcMixDKind<KeyType>*>(ptr);
}

template<>
struct unusedBits<MortonKey<unsigned>> : stl::integral_constant<unsigned, 2>
{
};
template<>
struct unusedBits<HilbertKey<unsigned>> : stl::integral_constant<unsigned, 2>
{
};

template<>
struct unusedBits<HilbertMixDKey<unsigned>> : stl::integral_constant<unsigned, 2>
{
};

template<>
struct unusedBits<MortonKey<unsigned long>> : stl::integral_constant<unsigned, 1>
{
};
template<>
struct unusedBits<HilbertKey<unsigned long>> : stl::integral_constant<unsigned, 1>
{
};

template<>
struct unusedBits<HilbertMixDKey<unsigned long>> : stl::integral_constant<unsigned, 1>
{
};

template<>
struct unusedBits<MortonKey<unsigned long long>> : stl::integral_constant<unsigned, 1>
{
};
template<>
struct unusedBits<HilbertKey<unsigned long long>> : stl::integral_constant<unsigned, 1>
{
};

template<>
struct unusedBits<HilbertMixDKey<unsigned long long>> : stl::integral_constant<unsigned, 1>
{
};

template<>
struct maxTreeLevel<MortonKey<unsigned>> : stl::integral_constant<unsigned, 10>
{
};
template<>
struct maxTreeLevel<HilbertKey<unsigned>> : stl::integral_constant<unsigned, 10>
{
};

template<>
struct maxTreeLevel<HilbertMixDKey<unsigned>> : stl::integral_constant<unsigned, 10>
{
};

template<>
struct maxTreeLevel<MortonKey<unsigned long>> : stl::integral_constant<unsigned, 21>
{
};
template<>
struct maxTreeLevel<HilbertKey<unsigned long>> : stl::integral_constant<unsigned, 21>
{
};

template<>
struct maxTreeLevel<HilbertMixDKey<unsigned long>> : stl::integral_constant<unsigned, 21>
{
};

template<>
struct maxTreeLevel<MortonKey<unsigned long long>> : stl::integral_constant<unsigned, 21>
{
};
template<>
struct maxTreeLevel<HilbertKey<unsigned long long>> : stl::integral_constant<unsigned, 21>
{
};

template<>
struct maxTreeLevel<HilbertMixDKey<unsigned long long>> : stl::integral_constant<unsigned, 21>
{
};

//! @brief Meta function to detect Morton key types
template<class KeyType>
struct IsMorton : std::bool_constant<std::is_same_v<KeyType, MortonKey<typename KeyType::ValueType>>>
{
};

//! @brief Meta function to detect Hilbert key types
template<class KeyType>
struct IsHilbert : std::bool_constant<std::is_same_v<KeyType, HilbertKey<typename KeyType::ValueType>>>
{
};

//! @brief Meta function to detect Mixed 1D Hilbert key types
template<class KeyType>
struct IsHilbertMixD : std::bool_constant<std::is_same_v<KeyType, HilbertMixDKey<typename KeyType::ValueType>>>
{
};

//! @brief Key encode overload for Morton keys
template<class KeyType>
HOST_DEVICE_FUN inline std::enable_if_t<IsMorton<KeyType>{}, KeyType> iSfcKey(unsigned ix, unsigned iy, unsigned iz)
{
    return KeyType{iMorton<typename KeyType::ValueType>(ix, iy, iz)};
}

//! @brief Key encode overload for Hilbert keys
template<class KeyType>
HOST_DEVICE_FUN inline std::enable_if_t<IsHilbert<KeyType>{}, KeyType> iSfcKey(unsigned ix, unsigned iy, unsigned iz)
{
    return KeyType{iHilbert<typename KeyType::ValueType>(ix, iy, iz)};
}

//! @brief Key encode overload for Mixed Hilbert keys
template<class KeyType>
HOST_DEVICE_FUN inline std::enable_if_t<IsHilbertMixD<KeyType>{}, KeyType>
iSfcMixDKey(unsigned ix, unsigned iy, unsigned iz, unsigned bx, unsigned by, unsigned bz)
{
    return KeyType{iHilbertMixD<typename KeyType::ValueType>(ix, iy, iz, bx, by, bz)};
}

template<class KeyType, class T>
HOST_DEVICE_FUN inline KeyType sfc3D(T x, T y, T z, T xmin, T ymin, T zmin, T mx, T my, T mz)
{
    constexpr int mcoord = (1u << maxTreeLevel<typename KeyType::ValueType>{}) - 1;

    int ix = std::floor(x * mx) - xmin * mx;
    int iy = std::floor(y * my) - ymin * my;
    int iz = std::floor(z * mz) - zmin * mz;

    ix = stl::min(ix, mcoord);
    iy = stl::min(iy, mcoord);
    iz = stl::min(iz, mcoord);

    assert(ix >= 0);
    assert(iy >= 0);
    assert(iz >= 0);

    return iSfcKey<KeyType>(ix, iy, iz);
}

template<class KeyType, class T>
HOST_DEVICE_FUN inline KeyType
sfcMixD(T x, T y, T z, T xmin, T ymin, T zmin, T mx, T my, T mz, unsigned bx, unsigned by, unsigned bz)
{
    const int mcoord_x = (1u << bx) - 1;
    const int mcoord_y = (1u << by) - 1;
    const int mcoord_z = (1u << bz) - 1;

    int ix = std::floor(x * mx) - xmin * mx;
    int iy = std::floor(y * my) - ymin * my;
    int iz = std::floor(z * mz) - zmin * mz;

    ix = stl::min(ix, mcoord_x);
    iy = stl::min(iy, mcoord_y);
    iz = stl::min(iz, mcoord_z);

    assert(ix >= 0);
    assert(iy >= 0);
    assert(iz >= 0);

    return iSfcMixDKey<KeyType>(ix, iy, iz, bx, by, bz);
}

/*! @brief Calculates a Hilbert key for a 3D point within the specified box
 *
 * @tparam    KeyType  32- or 64-bit Morton or Hilbert key type.
 * @param[in] x,y,z    input coordinates within the unit cube [0,1]^3
 * @param[in] box      bounding for coordinates
 * @return             the SFC key
 *
 * Note: -KeyType needs to be specified explicitly.
 *       -not specifying an unsigned type results in a compilation error
 */
template<class KeyType, class T>
HOST_DEVICE_FUN inline KeyType sfc3D(T x, T y, T z, const Box<T>& box)
{
    constexpr unsigned cubeLength = (1u << maxTreeLevel<typename KeyType::ValueType>{});

    return sfc3D<KeyType>(x, y, z, box.xmin(), box.ymin(), box.zmin(), cubeLength * box.ilx(), cubeLength * box.ily(),
                          cubeLength * box.ilz());
}

/*! @brief Calculates a MixD Hilbert key for a 3D point within the specified box
 *
 * @tparam    KeyType  32- or 64-bit Morton or Hilbert key type.
 * @param[in] x,y,z    input coordinates within the unit cube [0,1]^3
 * @param[in] box      bounding for coordinates
 * @return             the SFC key
 *
 * Note: -KeyType needs to be specified explicitly.
 *       -not specifying an unsigned type results in a compilation error
 */
template<class KeyType, class T>
HOST_DEVICE_FUN inline KeyType sfcMixD(T x, T y, T z, const Box<T>& box, unsigned bx, unsigned by, unsigned bz)
{
    // Scale cubeLength_x, cubeLength_y, cubeLength_z based on bx, by, bz
    // IM: bx, by and bz could be scaled based on the extents of the x, y, z dimensions
    //     however, this would probably make sense only if the box is not cubical and
    //     the points are uniformly distributed within the box.
    //     For now keep the bx, by and bz as arguments.
    assert(bx <= maxTreeLevel<typename KeyType::ValueType>{});
    assert(by <= maxTreeLevel<typename KeyType::ValueType>{});
    assert(bz <= maxTreeLevel<typename KeyType::ValueType>{});
    const unsigned cubeLength_x = (1u << bx);
    const unsigned cubeLength_y = (1u << by);
    const unsigned cubeLength_z = (1u << bz);

    return sfcMixD<KeyType>(x, y, z, box.xmin(), box.ymin(), box.zmin(), cubeLength_x * box.ilx(),
                            cubeLength_y * box.ily(), cubeLength_z * box.ilz(), bx, by, bz);
}

//! @brief decode a Morton key
template<class KeyType>
HOST_DEVICE_FUN inline std::enable_if_t<IsMorton<KeyType>{}, util::tuple<unsigned, unsigned, unsigned>>
decodeSfc(KeyType key)
{
    return decodeMorton<typename KeyType::ValueType>(key);
}

//! @brief decode a Hilbert key
template<class KeyType>
HOST_DEVICE_FUN inline std::enable_if_t<IsHilbert<KeyType>{}, util::tuple<unsigned, unsigned, unsigned>>
decodeSfc(KeyType key)
{
    return decodeHilbert<typename KeyType::ValueType>(key);
}

//! @brief create and integer box from Morton keys
template<class KeyType>
HOST_DEVICE_FUN inline std::enable_if_t<IsMorton<KeyType>{}, IBox> sfcIBox(KeyType keyStart, unsigned level) noexcept
{
    return mortonIBox<typename KeyType::ValueType>(keyStart, level);
}

//! @brief create and integer box from Hilbert keys
template<class KeyType>
HOST_DEVICE_FUN inline std::enable_if_t<IsHilbert<KeyType>{}, IBox> sfcIBox(KeyType keyStart, unsigned level) noexcept
{
    return hilbertIBox<typename KeyType::ValueType>(keyStart, level);
}

template<class KeyType>
HOST_DEVICE_FUN inline std::enable_if_t<IsHilbertMixD<KeyType>{}, IBox>
sfcIBox(KeyType keyStart, unsigned level, unsigned bx, unsigned by, unsigned bz) noexcept
{
    return hilbertMixDIBox<typename KeyType::ValueType>(keyStart, level, bx, by, bz);
}

//! @brief convenience overload
template<class KeyType>
HOST_DEVICE_FUN inline IBox sfcIBox(KeyType keyStart, KeyType keyEnd) noexcept
{
    return sfcIBox(keyStart, treeLevel(keyEnd - keyStart));
}

//! @brief convenience overload
template<class KeyType>
HOST_DEVICE_FUN inline IBox sfcIBox(KeyType keyStart, KeyType keyEnd, unsigned bx, unsigned by, unsigned bz) noexcept
{
    return sfcIBox(keyStart, maxTreeLevel<KeyType>{} - treeLevel(keyEnd - keyStart), bx, by, bz);
}

//! @brief Compute the smallest octree node in placeholder-bit format that contains the given floating point box
template<class KeyType, class T>
HOST_DEVICE_FUN inline KeyType commonNodePrefix(Vec3<T> center, Vec3<T> size, const cstone::Box<T>& box)
{
    KeyType lowerKey = cstone::sfc3D<KeyType>(center[0] - size[0], center[1] - size[1], center[2] - size[2], box);
    KeyType upperKey = cstone::sfc3D<KeyType>(center[0] + size[0], center[1] + size[1], center[2] + size[2], box);

    unsigned level  = commonPrefix(lowerKey, upperKey) / 3;
    KeyType nodeKey = enclosingBoxCode(lowerKey, level);

    return KeyType(encodePlaceholderBit(nodeKey.value(), 3 * level));
}

/*! @brief returns the smallest Hilbert key contained in the shifted box
 *
 * @tparam KeyType  32- or 64-bit unsigned integer
 * @param ibox      cubic integer coordinate box, edge length is a power of 2
 * @param dx        x-shift, in units of the ibox edge length
 * @param dy        y-shift, in units of the ibox edge length
 * @param dz        z-shift, in units of the ibox edge length
 * @return          the smallest key part of ibox shifted by (dx, dy, dz)
 */
template<class KeyType>
HOST_DEVICE_FUN inline KeyType sfcNeighbor(const IBox& ibox, unsigned level, int dx, int dy, int dz)
{
    constexpr unsigned pbcRange = 1u << maxTreeLevel<KeyType>{};

    unsigned shiftValue = ibox.xmax() - ibox.xmin();

    // lower corner of shifted box
    int x = pbcAdjust<pbcRange>(ibox.xmin() + dx * shiftValue);
    int y = pbcAdjust<pbcRange>(ibox.ymin() + dy * shiftValue);
    int z = pbcAdjust<pbcRange>(ibox.zmin() + dz * shiftValue);

    KeyType key = iSfcKey<KeyType>(x, y, z);

    return KeyType(enclosingBoxCode(key, level));
}

/*! @brief compute the SFC keys for the input coordinate arrays
 *
 * @tparam     T          float or double
 * @tparam     KeyType    HilbertKey or MortonKey
 * @param[in]  x          coordinate input arrays
 * @param[in]  y
 * @param[in]  z
 * @param[out] codeBegin  output for SFC keys
 * @param[in]  n          number of particles, size of input and output arrays
 * @param[in]  box        coordinate bounding box
 */
template<class T, class KeyType>
void computeSfcKeys(const T* x, const T* y, const T* z, KeyType* particleKeys, size_t n, const Box<T>& box)
{
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i)
    {
        if (particleKeys[i] != removeKey<KeyType>::value) { particleKeys[i] = sfc3D<KeyType>(x[i], y[i], z[i], box); }
        // std::cout << "[computeSfcKeys] particleKeys[" << i << "] = " << std::oct << particleKeys[i] << std::dec <<
        // std::endl;
    }
}

/*! @brief compute the mixed bit SFC keys for the input coordinate arrays
 *
 * @tparam     T          float or double
 * @tparam     KeyType    HilbertKey or MortonKey
 * @param[in]  x          coordinate input arrays
 * @param[in]  y
 * @param[in]  z
 * @param[out] codeBegin  output for SFC keys
 * @param[in]  n          number of particles, size of input and output arrays
 * @param[in]  box        coordinate bounding box
 * @param[in]  bx         number of bits to encode in x dimension
 * @param[in]  by         number of bits to encode in y dimension
 * @param[in]  bz         number of bits to encode in z dimension
 */
template<class T, class KeyType>
void computeSfcMixDKeys(const T* x,
                        const T* y,
                        const T* z,
                        KeyType* particleKeys,
                        size_t n,
                        const Box<T>& box,
                        unsigned bx,
                        unsigned by,
                        unsigned bz)
{
    // std::cout << "[computeSfcMixDKeys] box = " << box.xmin() << " " << box.xmax() << " " << box.ymin() << " " << box.ymax() << " "
    //           << box.zmin() << " " << box.zmax() << std::endl;
#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i)
    {
        if (particleKeys[i] != removeKey<KeyType>::value)
        {
            particleKeys[i] = sfcMixD<KeyType>(x[i], y[i], z[i], box, bx, by, bz);
        }
        // std::cout << "[computeSfcMixDKeys] particleKeys[" << i << "] = " << std::oct << particleKeys[i] << std::dec
        // << std::endl;
    }
}

} // namespace cstone
