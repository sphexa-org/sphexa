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

    std::array<unsigned, 3> bits{bx, by, bz};
    std::array<int, 3> permutation{0, 1, 2};
    std::sort(permutation.begin(), permutation.end(), [&bits](int i, int j) { return bits[i] > bits[j]; });
    std::array<KeyType, 3> coordinates{px, py, pz};
    std::array<KeyType, 3> sorted_coordinates{coordinates[permutation[0]], coordinates[permutation[1]],
                                               coordinates[permutation[2]]};
    std::sort(bits.begin(), bits.end(), std::greater<unsigned>{});

    if (bits[0] > bits[1]) // 1 dim has more bits than the other 2 dims, add 1D levels
    {
        const int n = bits[0] - bits[1];
        // add n 1D levels and add to key (trivial)
        for (int i{0}; i < n; ++i)
        {
            const auto processes_bit_index = bits[0] - i - 1;
            key |= ((sorted_coordinates[0] >> processes_bit_index) & 1) << (3 * processes_bit_index);
            // IM: Should it be 00? for x, 0?0 for y and ?00 for z?
        }
        const KeyType mask = (static_cast<KeyType>(1) << bits[1]) - 1;
        sorted_coordinates[0] &= mask;
        bits[0] -= n;
        // now we have bits[0] == bits[1]
    }

    if (bits[1] > bits[2]) // 2 dims have more bits than the 3rd, add 2D levels
    {
        const int n = bits[1] - bits[2];
        // encode n 2D levels with 2D-Hilbert and add it to the key
        // 2D key needs to be computed only for n bits
        const KeyType key_2D =
            iHilbert2D<KeyType>(sorted_coordinates[0] >> bits[2], sorted_coordinates[1] >> bits[2], n);
        // IM: Check if we want to the 2D key together or break it from 2 bits per level to 3 bits per level
        // key |= key_2D << (3 * bits[2]);
        // or below
        for (int i{0}; i < n; ++i)
        {
            const auto processes_2D_key_bit_index     = n - 1 - i;
            const auto processes_coordinate_bit_index = bits[1] - 1 - i;
            key |= ((key_2D >> (2 * processes_2D_key_bit_index)) & 3) << (3 * processes_coordinate_bit_index);
        }
        // remove n bits from sorted_coordinates[0] and sorted_coordinates[1]
        const KeyType mask = (1u << bits[2]) - 1;
        sorted_coordinates[0] &= mask;
        sorted_coordinates[1] &= mask;
        bits[0] -= n;
        bits[1] -= n;
        // now we have bits[0] == bits[1] == bits[2]
    }

    // Assert that the 3D coordinates of the 2 largest dimensions are smaller than the allowed range of the min
    // dimension to ensure that the first 3 * (bits[0] - bits[2]) bits are 0
    assert(sorted_coordinates[0] < (1u << bits[2]));
    assert(sorted_coordinates[1] < (1u << bits[2]));

    // encode remaining bits[0] == min(bx,by,bz) 3D levels or octal digits with 3D-Hilbert and add to key
    const KeyType key_3D = iHilbert<KeyType>(sorted_coordinates[0], sorted_coordinates[1], sorted_coordinates[2]);
    key |= key_3D;
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
    std::array<unsigned, 3> bits{bx, by, bz};
    std::array<int, 3> permutation{0, 1, 2};
    std::sort(permutation.begin(), permutation.end(), [&bits](int i, int j) { return bits[i] > bits[j]; });
    std::sort(bits.begin(), bits.end(), std::greater<unsigned>{});

    std::array<KeyType, 3> coordinates{0, 0, 0};

    if (bits[0] > bits[1]) // 1 dim has more bits than the other 2 dims, add 1D levels
    {
        const int n = bits[0] - bits[1];
        for (int i{0}; i < n; ++i)
        {
            const auto processes_coordinate_bit_index = bits[0] - 1 - i;
            coordinates[0] |= ((key >> (3 * processes_coordinate_bit_index)) & static_cast<KeyType>(1)) << processes_coordinate_bit_index;
        }
        key &= (static_cast<KeyType>(1) << (3 * bits[1])) - 1;
    }
    if (bits[1] > bits[2]) // 2 dims have more bits than the 3rd, add 2D levels
    {
        const int n = bits[1] - bits[2];
        // const auto key_2D  = key >> (3 * bits[2]);
        KeyType key_2D{};
        for (int i{}; i < n; ++i)
        {
            const auto processes_2D_key_bit_index     = n - 1 - i;
            const auto processes_coordinate_bit_index = bits[1] - 1 - i;
            key_2D |= ((key >> (3 * processes_coordinate_bit_index)) & 3) << (2 * processes_2D_key_bit_index);
        }
        const auto pair_2D = decodeHilbert2D<KeyType>(key_2D, bits[1] - bits[2]);
        coordinates[0] |= (get<0>(pair_2D) & ((1u << n) - 1)) << bits[2];
        coordinates[1] |= (get<1>(pair_2D) & ((1u << n) - 1)) << bits[2];
        key &= (1u << (3 * bits[2])) - 1;
    }

    const auto pair_3D = decodeHilbert<KeyType>(key);
    coordinates[0] |= get<0>(pair_3D);
    coordinates[1] |= get<1>(pair_3D);
    coordinates[2] |= get<2>(pair_3D);

    std::array<unsigned, 3> return_coordinates{0, 0, 0};
    return_coordinates[permutation[0]] = coordinates[0];
    return_coordinates[permutation[1]] = coordinates[1];
    return_coordinates[permutation[2]] = coordinates[2];

    return {return_coordinates[0], return_coordinates[1], return_coordinates[2]};
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
    auto [ix, iy, iz]           = decodeHilbert(keyStart);
    // std::cout << "keyStart (octal): " << std::oct << keyStart << std::dec << std::endl;
    // std::cout << "level: " << level << std::endl;
    // std::cout << "cubeLength: " << cubeLength << std::endl;
    // std::cout << "mask: " << std::bitset<32>(mask) << std::endl;
    // std::cout << "[before mask] ix (octal): " << std::oct << ix << " iy (octal): " << iy << " iz (octal): " << iz
    //           << std::dec << std::endl;

    // round integer coordinates down to corner closest to origin
    ix &= mask;
    iy &= mask;
    iz &= mask;
    // std::cout << "[after mask]  ix (octal): " << std::oct << ix << " iy (octal): " << iy << " iz (octal): " << iz
    //           << std::dec << std::endl;
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
    std::array<unsigned, 3> bits{bx, by, bz};
    std::sort(bits.begin(), bits.end());
    for (unsigned i{1}; i <= maxTreeLevel<KeyType>(); ++i)
    {
        const auto shifted_key = key >> (3 * (i - 1));
        const auto last_key_digit_of_shifted_key = shifted_key & 7u;
        if (i <= bits[0])
        {
            continue;
        }
        else if (i <= bits[1])
        {
            if (last_key_digit_of_shifted_key > 3u) {
                return false;
            }
        }
        else if (i <= bits[2])
        {
            if (last_key_digit_of_shifted_key > 1u) {
                return false;
            }
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
    // if constexpr (std::is_same_v<KeyType, unsigned>)
    // {
    //     std::cout << "[hilbertMixDIBox] KeyType is unsigned" << std::endl;
    // } else if constexpr (std::is_same_v<KeyType, unsigned long>)
    // {
    //     std::cout << "[hilbertMixDIBox] KeyType is unsigned long" << std::endl;
    // } else if constexpr (std::is_same_v<KeyType, unsigned long long>)
    // {
    //     std::cout << "[hilbertMixDIBox] KeyType is unsigned long long" << std::endl;
    // } else
    // {
    //     static_assert(std::is_same_v<KeyType, unsigned>, "KeyType must be unsigned");
    // }
    // std::cout << "[hilbertMixDIBox] keyStart (octal): " << std::oct << keyStart << std::dec << std::endl;
    // std::cout << "[hilbertMixDIBox] level: " << level << std::endl;
    // std::cout << "[hilbertMixDIBox] bx: " << bx << " by: " << by << " bz: " << bz << std::endl;
    assert(level <= maxTreeLevel<KeyType>{});
    auto is_valid_key = isValidHilbertMixDKey(keyStart, bx, by, bz);
    if (!is_valid_key)
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

    // std::cout << "cubeLengthX: " << cubeLengthX << " cubeLengthY: " << cubeLengthY << " cubeLengthZ: " << cubeLengthZ << std::endl;
    // std::cout << "cubeLength: " << cubeLength << std::endl;
    // std::cout << "mask: " << std::bitset<32>(mask) << std::endl;
    // std::cout << "[hilbertMixDIBox] [before mask] ix (octal): " << std::oct << ix << " iy (octal): " << iy
    //           << " iz (octal): " << iz << std::dec << std::endl;
    // std::cout << "[hilbertMixDIBox] maskX: " << std::oct << maskX << " maskY: " << std::oct << maskY << " maskZ: " << std::oct << maskZ << std::dec << std::endl;
    // std::cout << "[before mask] ix (octal): " << std::oct << ix << " iy (octal): " << iy << " iz (octal): " << iz
    //   << std::dec << std::endl;

    // round integer coordinates down to corner closest to origin
    ix &= maskX;
    iy &= maskY;
    iz &= maskZ;
    // std::cout << "[after mask]  ix (octal): " << std::oct << ix << " iy (octal): " << iy << " iz (octal): " << iz
    //   << std::dec << std::endl;
    return IBox(ix, ix + cubeLengthX, iy, iy + cubeLengthY, iz, iz + cubeLengthZ);
}

template<class KeyType>
HOST_DEVICE_FUN constexpr unsigned treeLevelMixD(KeyType codeRange, unsigned bx, unsigned by, unsigned bz) noexcept
{
    std::array<unsigned, 3> bits{bx, by, bz};
    std::sort(bits.begin(), bits.end());
    unsigned level{0};
    KeyType codeRangeLevel{1};
    while (codeRange > codeRangeLevel)
    {
        if (level < bits[0]) { codeRangeLevel <<= 3; }
        else if (level < bits[1]) { codeRangeLevel <<= 2; }
        else { codeRangeLevel <<= 1; }
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
    std::cout << "Diff: " << diff << " oct: " << std::oct << diff << std::dec
              << " bin: " << std::bitset<sizeof(KeyType) * 8>(diff) << std::endl;
    return hilbertMixDIBox(keyStart, treeLevelMixD(diff, bx, by, bz), bx, by, bz);
}

} // namespace cstone
