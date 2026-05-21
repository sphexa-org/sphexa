/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  3D Hilbert encoding/decoding in 32- and 64-bit
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * This code is based on the implementation of the Hilbert curve presented in:
 *
 * Yohei Miki, Masayuki Umemura
 * GOTHIC: Gravitational oct-tree code accelerated by hierarchical time step controlling
 * https://doi.org/10.1016/j.newast.2016.10.007
 *
 * The 2D Hilbert curve  code is based on the book by Henry S. Warren
 * https://learning.oreilly.com/library/view/hackers-delight-second
 */

#pragma once

#include <iostream>

#include "cstone/util/tuple.hpp"
#include "morton.hpp"

namespace cstone
{

#if defined(__CUDACC__) || defined(__HIPCC__)
__device__ static unsigned mortonToHilbertDevice[8] = {0, 1, 3, 2, 7, 6, 4, 5};
#endif

/*! @brief compute the Hilbert key for a 3D point of integer coordinates
 *
 * @tparam     KeyType   32- or 64-bit unsigned integer
 * @param[in]  px,py,pz  input coordinates in [0:2^maxTreeLevel<KeyType>{}]
 * @return               the Hilbert key
 */
template<class KeyType>
constexpr HOST_DEVICE_FUN inline std::enable_if_t<std::is_unsigned_v<KeyType>, KeyType>
iHilbert(unsigned px, unsigned py, unsigned pz, int order = maxTreeLevel<KeyType>{}) noexcept
{
    assert(px < (1u << order));
    assert(py < (1u << order));
    assert(pz < (1u << order));

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
    constexpr unsigned mortonToHilbert[8] = {0, 1, 3, 2, 7, 6, 4, 5};
#endif

    KeyType key = 0;

    for (int level = order - 1; level >= 0; --level)
    {
        unsigned xi = (px >> level) & 1u;
        unsigned yi = (py >> level) & 1u;
        unsigned zi = (pz >> level) & 1u;

        // append 3 bits to the key
        unsigned octant = (xi << 2) | (yi << 1) | zi;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        key = (key << 3) + mortonToHilbertDevice[octant];
#else
        key = (key << 3) + mortonToHilbert[octant];
#endif

        // turn px, py and pz
        px ^= -(xi & ((!yi) | zi));
        py ^= -((xi & (yi | zi)) | (yi & (!zi)));
        pz ^= -((xi & (!yi) & (!zi)) | (yi & (!zi)));

        if (zi)
        {
            // cyclic rotation
            unsigned pt = px;
            px          = py;
            py          = pz;
            pz          = pt;
        }
        else if (!yi)
        {
            // swap x and z
            unsigned pt = px;
            px          = pz;
            pz          = pt;
        }
    }

    return key;
}

template<class KeyType>
HOST_DEVICE_FUN std::enable_if_t<std::is_unsigned_v<KeyType>, KeyType>
iHilbert2D(unsigned px, unsigned py, int order = maxTreeLevel<KeyType>{}) noexcept;

/*! @brief compute the Hilbert key for a 3D point of integer coordinates
 *
 * @tparam     KeyType   32- or 64-bit unsigned integer
 * @param[in]  px        input x integer coordinate, in [0:2^bx]
 * @param[in]  py        input y integer coordinate, in [0:2^by]
 * @param[in]  pz        input z integer coordinate, in [0:2^bz]
 * @param[in]  bx        number of bits to encode in x dimension, in [0:maxTreelevel<KeyType>{}]
 * @param[in]  by        number of bits to encode in y dimension, in [0:maxTreelevel<KeyType>{}]
 * @param[in]  bz        number of bits to encode in z dimension, in [0:maxTreelevel<KeyType>{}]
 * @return               the Hilbert key
 *
 * Example box with (Lx, Ly, Lz) = (8,4,1):
 *  The longest dimension will get the max number of bits per dimension maxTreelevel<KeyType>{},
 *  i.e 10 bits if KeyType is 32-bit. The bits in the other dimensions are reduced by 1 for each
 *  factor of 2 that the box is shorter in that dimension than the longest. For the example box,
 *  (bx, by, bz) will be (10, 9, 7)
 */
template<class KeyType>
constexpr HOST_DEVICE_FUN inline std::enable_if_t<std::is_unsigned_v<KeyType>, KeyType>
iHilbertMixD(unsigned px, unsigned py, unsigned pz, unsigned bx, unsigned by, unsigned bz) noexcept
{
    assert(px < (1u << bx));
    assert(py < (1u << by));
    assert(pz < (1u << bz));
    assert(bx <= maxTreeLevel<KeyType>{} && by <= maxTreeLevel<KeyType>{} && bz <= maxTreeLevel<KeyType>{});

    KeyType key = 0;

    // Sort bits[] descending while tracking permutation[] — 3-element sort network (GPU-friendly)
    unsigned bits[3]   = {bx, by, bz};
    int permutation[3] = {0, 1, 2};
    if (bits[0] < bits[1])
    {
        unsigned t     = bits[0];
        bits[0]        = bits[1];
        bits[1]        = t;
        int tp         = permutation[0];
        permutation[0] = permutation[1];
        permutation[1] = tp;
    }
    if (bits[0] < bits[2])
    {
        unsigned t     = bits[0];
        bits[0]        = bits[2];
        bits[2]        = t;
        int tp         = permutation[0];
        permutation[0] = permutation[2];
        permutation[2] = tp;
    }
    if (bits[1] < bits[2])
    {
        unsigned t     = bits[1];
        bits[1]        = bits[2];
        bits[2]        = t;
        int tp         = permutation[1];
        permutation[1] = permutation[2];
        permutation[2] = tp;
    }

    KeyType coordinates[3]       = {px, py, pz};
    KeyType sortedCoordinates[3] = {coordinates[permutation[0]], coordinates[permutation[1]],
                                    coordinates[permutation[2]]};

    if (bits[0] > bits[1]) // 1 dim has more bits than the other 2 dims, add 1D levels
    {
        const int n = bits[0] - bits[1];
        // add n 1D levels and add to key (trivial)
        for (int i{0}; i < n; ++i)
        {
            const auto processesBitIndex = bits[0] - i - 1;
            key |= static_cast<KeyType>(static_cast<KeyType>(sortedCoordinates[0] >> processesBitIndex) & 1)
                   << (3 * processesBitIndex);
            // IM: Should it be 00? for x, 0?0 for y and ?00 for z?
        }
        const KeyType mask = (static_cast<KeyType>(1) << bits[1]) - 1;
        sortedCoordinates[0] &= mask;
        bits[0] -= n;
        // now we have bits[0] == bits[1]
    }

    if (bits[1] > bits[2]) // 2 dims have more bits than the 3rd, add 2D levels
    {
        const int n = bits[1] - bits[2];
        // encode n 2D levels with 2D-Hilbert and add it to the key
        // 2D key needs to be computed only for n bits
        const KeyType key2D = iHilbert2D<KeyType>(sortedCoordinates[0] >> bits[2], sortedCoordinates[1] >> bits[2], n);
        // IM: Check if we want to the 2D key together or break it from 2 bits per level to 3 bits per level
        // key |= key2D << (3 * bits[2]);
        // or below
        for (int i{0}; i < n; ++i)
        {
            const auto processes2DKeyBitIndex      = n - 1 - i;
            const auto processesCoordinateBitIndex = bits[1] - 1 - i;
            key |= static_cast<KeyType>(static_cast<KeyType>(key2D >> (2 * processes2DKeyBitIndex)) & 3)
                   << (3 * processesCoordinateBitIndex);
        }
        // remove n bits from sortedCoordinates[0] and sortedCoordinates[1]
        const KeyType mask = (static_cast<KeyType>(1) << bits[2]) - 1;
        sortedCoordinates[0] &= mask;
        sortedCoordinates[1] &= mask;
        bits[0] -= n;
        bits[1] -= n;
        // now we have bits[0] == bits[1] == bits[2]
    }

    // Assert that the 3D coordinates of the 2 largest dimensions are smaller than the allowed range of the min
    // dimension to ensure that the first 3 * (bits[0] - bits[2]) bits are 0
    assert(sortedCoordinates[0] < (static_cast<KeyType>(1) << bits[2]));
    assert(sortedCoordinates[1] < (static_cast<KeyType>(1) << bits[2]));

    // encode remaining bits[0] == min(bx,by,bz) 3D levels or octal digits with 3D-Hilbert and add to key
    const KeyType key3D = iHilbert<KeyType>(sortedCoordinates[0], sortedCoordinates[1], sortedCoordinates[2]);
    key |= key3D;
    // Example for (bx,by,bz) = (10,9,7): 1D,2D,2D,3D*7

    return key;
}

/*! @brief compute the Hilbert key for a 2D point of integer coordinates
 *
 * @tparam     KeyType   32- or 64-bit unsigned integer
 * @param[in]  px,py  input coordinates in [0:2^maxTreeLevel<KeyType>{}]
 * @return               the Hilbert key
 */

template<class KeyType>
HOST_DEVICE_FUN std::enable_if_t<std::is_unsigned_v<KeyType>, KeyType>
iHilbert2D(unsigned px, unsigned py, int order) noexcept
{
    assert(px < (1u << maxTreeLevel<KeyType>{}));
    assert(py < (1u << maxTreeLevel<KeyType>{}));

    unsigned xi, yi;
    unsigned temp;
    KeyType key = 0;

    for (int level = order - 1; level >= 0; level--)
    {
        xi = (px >> level) & 1u; // Get bit level of x.
        yi = (py >> level) & 1u; // Get bit level of y.

        if (yi == 0)
        {
            temp = px;           // Swap x and y and,
            px   = py ^ (-xi);   // if xi = 1,
            py   = temp ^ (-xi); // complement them.
        }
        key = 4 * key + 2 * xi + (xi ^ yi); // Append two bits to key.
    }
    return key;
}

//! @brief inverse function of iHilbert
template<class KeyType>
HOST_DEVICE_FUN inline util::tuple<unsigned, unsigned, unsigned>
decodeHilbert(KeyType key, unsigned order = maxTreeLevel<KeyType>{}) noexcept
{
    unsigned px = 0;
    unsigned py = 0;
    unsigned pz = 0;

    for (unsigned level = 0; level < order; ++level)
    {
        unsigned octant   = (key >> (3 * level)) & 7u;
        const unsigned xi = octant >> 2u;
        const unsigned yi = (octant >> 1u) & 1u;
        const unsigned zi = octant & 1u;

        if (yi ^ zi)
        {
            // cyclic rotation
            unsigned pt = px;
            px          = pz;
            pz          = py;
            py          = pt;
        }
        else if ((!xi & !yi & !zi) || (xi & yi & zi))
        {
            // swap x and z
            unsigned pt = px;
            px          = pz;
            pz          = pt;
        }

        // turn px, py and pz
        unsigned mask = (1 << level) - 1;
        px ^= mask & (-(xi & (yi | zi)));
        py ^= mask & (-((xi & ((!yi) | (!zi))) | ((!xi) & yi & zi)));
        pz ^= mask & (-((xi & (!yi) & (!zi)) | (yi & zi)));

        // append 1 bit to the positions
        px |= (xi << level);
        py |= ((xi ^ yi) << level);
        pz |= ((yi ^ zi) << level);
    }

    return {px, py, pz};
}

// Lam and Shapiro inverse function of hilbert
template<class KeyType>
HOST_DEVICE_FUN inline util::tuple<unsigned, unsigned>
decodeHilbert2D(KeyType key, unsigned order = maxTreeLevel<KeyType>{}) noexcept
{
    unsigned sa, sb;
    unsigned x = 0, y = 0, temp = 0;

    for (unsigned level = 0; level < 2 * order; level += 2)
    {
        // Get bit level+1 of key.
        sa = (key >> (level + 1)) & 1;
        // Get bit level of key.
        sb = (key >> level) & 1;
        if ((sa ^ sb) == 0)
        {
            // If sa,sb = 00 or 11,
            temp = x;
            // swap x and y,
            x = y ^ (-sa);
            // and if sa = 1,
            y = temp ^ (-sa);
            // complement them.
        }
        x = (x >> 1) | (sa << 31);        // Prepend sa to x and
        y = (y >> 1) | ((sa ^ sb) << 31); // (sa ^ sb) to y.
    }
    unsigned px = x >> (32 - order);
    // Right-adjust x and y
    unsigned py = y >> (32 - order);
    // and return them to
    return {px, py};
}

//! @brief inverse function of iHilbertMixD
template<class KeyType>
HOST_DEVICE_FUN inline util::tuple<unsigned, unsigned, unsigned>
decodeHilbertMixD(KeyType key, unsigned bx, unsigned by, unsigned bz) noexcept
{
    // Sort bits[] descending while tracking permutation[] — 3-element sort network (GPU-friendly)
    unsigned bits[3]   = {bx, by, bz};
    int permutation[3] = {0, 1, 2};
    if (bits[0] < bits[1])
    {
        unsigned t     = bits[0];
        bits[0]        = bits[1];
        bits[1]        = t;
        int tp         = permutation[0];
        permutation[0] = permutation[1];
        permutation[1] = tp;
    }
    if (bits[0] < bits[2])
    {
        unsigned t     = bits[0];
        bits[0]        = bits[2];
        bits[2]        = t;
        int tp         = permutation[0];
        permutation[0] = permutation[2];
        permutation[2] = tp;
    }
    if (bits[1] < bits[2])
    {
        unsigned t     = bits[1];
        bits[1]        = bits[2];
        bits[2]        = t;
        int tp         = permutation[1];
        permutation[1] = permutation[2];
        permutation[2] = tp;
    }

    KeyType coordinates[3] = {0, 0, 0};

    if (bits[0] > bits[1]) // 1 dim has more bits than the other 2 dims, add 1D levels
    {
        const int n = bits[0] - bits[1];
        for (int i{0}; i < n; ++i)
        {
            const auto processesCoordinateBitIndex = bits[0] - 1 - i;
            coordinates[0] |= static_cast<KeyType>(static_cast<KeyType>(key >> (3 * processesCoordinateBitIndex)) &
                                                   static_cast<KeyType>(1))
                              << processesCoordinateBitIndex;
        }
        key &= (static_cast<KeyType>(1) << (3 * bits[1])) - 1;
    }
    if (bits[1] > bits[2]) // 2 dims have more bits than the 3rd, add 2D levels
    {
        const int n = bits[1] - bits[2];
        // const auto key2D  = key >> (3 * bits[2]);
        KeyType key2D{};
        for (int i{}; i < n; ++i)
        {
            const auto processes2DKeyBitIndex      = n - 1 - i;
            const auto processesCoordinateBitIndex = bits[1] - 1 - i;
            key2D |= static_cast<KeyType>(static_cast<KeyType>(key >> (3 * processesCoordinateBitIndex)) & 3)
                     << (2 * processes2DKeyBitIndex);
        }
        const auto pair2D = decodeHilbert2D<KeyType>(key2D, bits[1] - bits[2]);
        coordinates[0] |= (get<0>(pair2D) & ((static_cast<KeyType>(1) << n) - 1)) << bits[2];
        coordinates[1] |= (get<1>(pair2D) & ((static_cast<KeyType>(1) << n) - 1)) << bits[2];
        key &= (static_cast<KeyType>(1) << (3 * bits[2])) - 1;
    }

    const auto pair3D = decodeHilbert<KeyType>(key);
    coordinates[0] |= get<0>(pair3D);
    coordinates[1] |= get<1>(pair3D);
    coordinates[2] |= get<2>(pair3D);

    KeyType returnCoordinates[3]      = {0, 0, 0};
    returnCoordinates[permutation[0]] = coordinates[0];
    returnCoordinates[permutation[1]] = coordinates[1];
    returnCoordinates[permutation[2]] = coordinates[2];

    return {returnCoordinates[0], returnCoordinates[1], returnCoordinates[2]};
}

//! @brief inverse function of iHilbert 32 bit only up to oder 16 but works at constant time.
template<class KeyType>
HOST_DEVICE_FUN inline util::tuple<unsigned, unsigned> decodeHilbert2DConstant(KeyType key) noexcept
{
    unsigned order = maxTreeLevel<KeyType>{};

    key = key | (0x55555555 << 2 * order); // Pad key on left with 01

    const unsigned sr = (key >> 1) & 0x55555555;                // (no change) groups.
    unsigned cs       = ((key & 0x55555555) + sr) ^ 0x55555555; // Compute complement & swap info in two-bit groups.
    // Parallel prefix xor op to propagate both complement
    // and swap info together from left to right (there is
    // no step "cs ^= cs >> 1", so in effect it computes
    // two independent parallel prefix operations on two
    // interleaved sets of sixteen bits).
    cs                  = cs ^ (cs >> 2);
    cs                  = cs ^ (cs >> 4);
    cs                  = cs ^ (cs >> 8);
    cs                  = cs ^ (cs >> 16);
    const unsigned swap = cs & 0x55555555;        // Separate the swap and
    const unsigned comp = (cs >> 1) & 0x55555555; // complement bits.

    unsigned t = (key & swap) ^ comp;          // Calculate x and y in
    key        = key ^ sr ^ t ^ (t << 1);      // the odd & even bit positions, resp.
    key        = key & ((1 << 2 * order) - 1); // Clear out any junk on the left (unpad).

    // Now "unshuffle" to separate the x and y bits.

    t   = (key ^ (key >> 1)) & 0x22222222;
    key = key ^ t ^ (t << 1);
    t   = (key ^ (key >> 2)) & 0x0C0C0C0C;
    key = key ^ t ^ (t << 2);
    t   = (key ^ (key >> 4)) & 0x00F000F0;
    key = key ^ t ^ (t << 4);
    t   = (key ^ (key >> 8)) & 0x0000FF00;
    key = key ^ t ^ (t << 8);

    unsigned px = key >> 16;    // Assign the two halves
    unsigned py = key & 0xFFFF; // of t to x and y.

    return {px, py};
}

/*! @brief compute the 3D integer coordinate box that contains the key range
 *
 * @tparam KeyType   32- or 64-bit unsigned integer
 * @param  keyStart  lower Hilbert key
 * @param  keyEnd    upper Hilbert key
 * @return           the integer box that contains the given key range
 */
template<class KeyType>
HOST_DEVICE_FUN IBox hilbertIBox(KeyType keyStart, unsigned level) noexcept
{
    assert(level <= maxTreeLevel<KeyType>{});
    constexpr unsigned maxCoord = 1u << maxTreeLevel<KeyType>{};
    unsigned cubeLength         = maxCoord >> level;
    unsigned mask               = ~(cubeLength - 1);

    auto [ix, iy, iz] = decodeHilbert(keyStart);

    // round integer coordinates down to corner closest to origin
    ix &= mask;
    iy &= mask;
    iz &= mask;

    return IBox(ix, ix + cubeLength, iy, iy + cubeLength, iz, iz + cubeLength);
}

//! @brief convenience wrapper
template<class KeyType>
HOST_DEVICE_FUN IBox hilbertIBoxKeys(KeyType keyStart, KeyType keyEnd) noexcept
{
    assert(keyStart <= keyEnd);
    return hilbertIBox(keyStart, treeLevel(keyEnd - keyStart));
}

template<class KeyType>
HOST_DEVICE_FUN bool isValidHilbertMixDKey(KeyType key, unsigned bx, unsigned by, unsigned bz) noexcept
{
    // Ascending 3-element sort network (GPU-friendly, no std::sort)
    unsigned bits[3] = {bx, by, bz};
    if (bits[0] > bits[1])
    {
        unsigned t = bits[0];
        bits[0]    = bits[1];
        bits[1]    = t;
    }
    if (bits[0] > bits[2])
    {
        unsigned t = bits[0];
        bits[0]    = bits[2];
        bits[2]    = t;
    }
    if (bits[1] > bits[2])
    {
        unsigned t = bits[1];
        bits[1]    = bits[2];
        bits[2]    = t;
    }
    for (unsigned i{1}; i <= maxTreeLevel<KeyType>(); ++i)
    {
        const KeyType shiftedKey               = key >> (3 * (i - 1));
        const KeyType lastKeyDigitOfShiftedKey = shiftedKey & 7u;
        if (i <= bits[0]) { continue; }
        else if (i <= bits[1])
        {
            if (lastKeyDigitOfShiftedKey > 3u) { return false; }
        }
        else if (i <= bits[2])
        {
            if (lastKeyDigitOfShiftedKey > 1u) { return false; }
        }
    }
    return true;
}

/*! @brief compute the 3D integer coordinate box that contains the key range
 *
 * @tparam KeyType   32- or 64-bit unsigned integer
 * @param  level     level from the right
 * @param  keyStart  lower Hilbert key
 * @param  keyEnd    upper Hilbert key
 * @return           the integer box that contains the given key range
 */
template<class KeyType>
HOST_DEVICE_FUN IBox hilbertMixDIBox(KeyType keyStart, unsigned level, unsigned bx, unsigned by, unsigned bz) noexcept
{
    assert(level <= maxTreeLevel<KeyType>{});
    auto isValidKey = isValidHilbertMixDKey(keyStart, bx, by, bz);
    if (!isValidKey)
    {
        return IBox(0, 0, 0, 0, 0, 0); // return empty box
    }
    unsigned cubeLengthX = 1u << std::min(bx, level);
    unsigned cubeLengthY = 1u << std::min(by, level);
    unsigned cubeLengthZ = 1u << std::min(bz, level);
    unsigned maskX       = ~(cubeLengthX - 1);
    unsigned maskY       = ~(cubeLengthY - 1);
    unsigned maskZ       = ~(cubeLengthZ - 1);
    auto [ix, iy, iz]    = decodeHilbertMixD<KeyType>(keyStart, bx, by, bz);

    // round integer coordinates down to corner closest to origin
    ix &= maskX;
    iy &= maskY;
    iz &= maskZ;
    return IBox(ix, ix + cubeLengthX, iy, iy + cubeLengthY, iz, iz + cubeLengthZ);
}

template<class KeyType>
HOST_DEVICE_FUN constexpr unsigned treeLevelMixD(KeyType codeRange, unsigned bx, unsigned by, unsigned bz) noexcept
{
    // Ascending 3-element sort network (GPU-friendly, no std::sort)
    unsigned bits[3] = {bx, by, bz};
    if (bits[0] > bits[1])
    {
        unsigned t = bits[0];
        bits[0]    = bits[1];
        bits[1]    = t;
    }
    if (bits[0] > bits[2])
    {
        unsigned t = bits[0];
        bits[0]    = bits[2];
        bits[2]    = t;
    }
    if (bits[1] > bits[2])
    {
        unsigned t = bits[1];
        bits[1]    = bits[2];
        bits[2]    = t;
    }
    unsigned level{0};
    KeyType codeRangeLevel{1};
    while (codeRange > codeRangeLevel)
    {
        if (level < bits[0]) { codeRangeLevel <<= 3; }
        else if (level < bits[1]) { codeRangeLevel <<= 2; }
        else
        {
            codeRangeLevel <<= 1;
        }
        level++;
    }
    return level;
}

//! @brief convenience wrapper
template<class KeyType>
HOST_DEVICE_FUN IBox
hilbertMixDIBoxKeys(KeyType keyStart, KeyType keyEnd, unsigned bx, unsigned by, unsigned bz) noexcept
{
    assert(keyStart < keyEnd);
    std::cout << "keyStart (octal): " << std::oct << keyStart << std::dec << std::endl;
    std::cout << "keyEnd (octal): " << std::oct << keyEnd << std::dec << std::endl;
    // keyEnd - keyStart needs to be power of 8
    // check if octree nodes work with
    // return hilbertMixDIBox(keyStart, treeLevel(keyEnd-keyStart), bx, by, bz);
    // what if keyStart 1377 and keyEnd 2000? -> shouldn't be a case like that because hilbertMixDIBoxKeys should only
    // take octree nodes' keys if keyStart is outside the bx, by, bz bounds, then the returned box edges should be
    // (0,0,0)
    KeyType diff{};
    for (KeyType i{keyStart}; i < keyEnd; i = increaseKey(i, 10, bx, by, bz))
    {
        diff++;
    }
    return hilbertMixDIBox(keyStart, treeLevelMixD(diff, bx, by, bz), bx, by, bz);
}

} // namespace cstone
