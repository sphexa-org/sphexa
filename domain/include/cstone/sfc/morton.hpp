/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  3D Morton encoding/decoding in 32- and 64-bit using the magic number method
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cassert>
#include <cstdint>     // for uint32_t and uint64_t
#include <type_traits> // for std::enable_if_t

#include "cstone/util/tuple.hpp"

#include "box.hpp"
#include "common.hpp"

namespace cstone
{

namespace detail
{

//! @brief Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
HOST_DEVICE_FUN
constexpr unsigned expandBits(unsigned v)
{
    v &= 0x000003ffu; // discard bit higher 10
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

/*! @brief Compacts a 30-bit integer into 10 bits by selecting only bits divisible by 3
 *         this inverts expandBits
 */
HOST_DEVICE_FUN
constexpr unsigned compactBits(unsigned v)
{
    v &= 0x09249249u;
    v = (v ^ (v >> 2u)) & 0x030c30c3u;
    v = (v ^ (v >> 4u)) & 0x0300f00fu;
    v = (v ^ (v >> 8u)) & 0xff0000ffu;
    v = (v ^ (v >> 16u)) & 0x000003ffu;
    return v;
}

//! @brief Expands a 21-bit integer into 63 bits by inserting 2 zeros after each bit.
HOST_DEVICE_FUN
constexpr uint64_t expandBits(uint64_t v)
{
    uint64_t x = v & 0x1fffffu; // discard bits higher 21
    x          = (x | x << 32u) & 0x001f00000000fffflu;
    x          = (x | x << 16u) & 0x001f0000ff0000fflu;
    x          = (x | x << 8u) & 0x100f00f00f00f00flu;
    x          = (x | x << 4u) & 0x10c30c30c30c30c3lu;
    x          = (x | x << 2u) & 0x1249249249249249lu;
    return x;
}

/*! @brief Compacts a 63-bit integer into 21 bits by selecting only bits divisible by 3
 *         this inverts expandBits
 */
HOST_DEVICE_FUN
constexpr uint64_t compactBits(uint64_t v)
{
    v &= 0x1249249249249249lu;
    v = (v ^ (v >> 2u)) & 0x10c30c30c30c30c3lu;
    v = (v ^ (v >> 4u)) & 0x100f00f00f00f00flu;
    v = (v ^ (v >> 8u)) & 0x001f0000ff0000fflu;
    v = (v ^ (v >> 16u)) & 0x001f00000000fffflu;
    v = (v ^ (v >> 32u)) & 0x00000000001ffffflu;
    return v;
}

} // namespace detail

/*! @brief Calculates a Morton code for a 3D point in integer coordinates
 *
 * @tparam    KeyType  32- or 64 bit unsigned integer
 * @param[in] ix,iy,iz input coordinates in [0:2^maxTreeLevel<KeyType>{}]
 */
template<class KeyType>
constexpr HOST_DEVICE_FUN std::enable_if_t<std::is_unsigned<KeyType>{}, KeyType>
iMorton(unsigned ix, unsigned iy, unsigned iz) noexcept
{
    assert(ix < (1u << maxTreeLevel<KeyType>{}));
    assert(iy < (1u << maxTreeLevel<KeyType>{}));
    assert(iz < (1u << maxTreeLevel<KeyType>{}));

    KeyType xx = detail::expandBits(KeyType(ix));
    KeyType yy = detail::expandBits(KeyType(iy));
    KeyType zz = detail::expandBits(KeyType(iz));

    // interleave the x, y, z components
    return xx * 4 + yy * 2 + zz;
}

/*! @brief Calculate morton code from n-level integer 3D coordinates
 *
 * @tparam KeyType   32- or 64-bit unsigned integer
 * @param ix,iy,iz   input integer box coordinates, must be in the range [0, 2^treeLevel-1]
 * @param treeLevel  octree subdivison level
 * @return           the morton code
 */
template<class KeyType>
constexpr HOST_DEVICE_FUN KeyType iMorton(unsigned ix, unsigned iy, unsigned iz, unsigned treeLevel)
{
    assert(treeLevel <= maxTreeLevel<KeyType>{});
    unsigned shifts = maxTreeLevel<KeyType>{} - treeLevel;
    return iMorton<KeyType>(ix << shifts, iy << shifts, iz << shifts);
}

//! @brief extract X component from a morton code
template<class KeyType>
constexpr HOST_DEVICE_FUN inline std::enable_if_t<std::is_unsigned<KeyType>{}, unsigned> idecodeMortonX(KeyType code)
{
    return detail::compactBits(code >> 2);
}

//! @brief extract Y component from a morton code
template<class KeyType>
constexpr HOST_DEVICE_FUN inline std::enable_if_t<std::is_unsigned<KeyType>{}, unsigned> idecodeMortonY(KeyType code)
{
    return detail::compactBits(code >> 1);
}

//! @brief extract Z component from a morton code
template<class KeyType>
constexpr HOST_DEVICE_FUN inline std::enable_if_t<std::is_unsigned<KeyType>{}, unsigned> idecodeMortonZ(KeyType code)
{
    return detail::compactBits(code);
}

//! @brief decode X,Y,Z components of a Morton key into a tuple
template<class KeyType>
HOST_DEVICE_FUN inline util::tuple<unsigned, unsigned, unsigned> decodeMorton(KeyType code) noexcept
{
    return {idecodeMortonX(code), idecodeMortonY(code), idecodeMortonZ(code)};
}

/*! @brief compute the 3D integer coordinate box that contains the key range
 *
 * @tparam KeyType   32- or 64-bit unsigned integer
 * @param keyStart   lower Morton key of the tree cell
 * @param level      octree subdivision level of the tree cell
 * @return           the integer box that contains the given key range
 */
template<class KeyType>
HOST_DEVICE_FUN IBox mortonIBox(KeyType keyStart, unsigned level) noexcept
{
    assert(level <= maxTreeLevel<KeyType>{});
    unsigned cubeLength = (1u << (maxTreeLevel<KeyType>{} - level));
    auto [ix, iy, iz]   = decodeMorton(keyStart);
    return IBox(ix, ix + cubeLength, iy, iy + cubeLength, iz, iz + cubeLength);
}

template<class KeyType>
HOST_DEVICE_FUN inline IBox mortonIBoxKeys(KeyType keyStart, KeyType keyEnd) noexcept
{
    assert(keyStart <= keyEnd);
    return mortonIBox(keyStart, treeLevel(keyEnd - keyStart));
}

} // namespace cstone
