/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Common operations on SFC keys that do not depend on the specific SFC used
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cassert>
#include <cmath>
#include <type_traits>

#include "cstone/primitives/clz.hpp"
#include "cstone/primitives/stl.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/util/tuple.hpp"

namespace cstone
{

/*! @brief normalize a floating point number in [0,1] to an integer in [0 : 2^(10 or 21)]
 *
 * @tparam KeyType  32-bit or 64-bit unsigned integer
 * @tparam T        float or double
 * @param  x        input floating point number in [0,1]
 * @return          x converted to an 10-bit or 21-bit integer
 *                  maximum return value is 1023 or 2097151
 *
 * Integer conversion happens with truncation as required for SFC code calculations
 */
template<class KeyType, class T>
HOST_DEVICE_FUN unsigned toNBitInt(T x)
{
    // spatial resolution in bits per dimension
    constexpr unsigned nBits = maxTreeLevel<KeyType>{};

    // [0,1] to [0:1024] and convert to integer (32-bit) or
    // [0,1] to [0:2097152] and convert to integer (64-bit)
    unsigned result = x * T(1u << nBits);
    return stl::min(result, (1u << nBits) - 1u);
}

/*! @brief normalize a floating point number in [0,1] to an integer in [0 : 2^(10 or 21)]
 *
 * @tparam KeyType  32-bit or 64-bit unsigned integer
 * @tparam T        float or double
 * @param  x        input floating point number in [0,1]
 * @return          x converted to an 10-bit or 21-bit integer
 *                  maximum return value is 1023 or 2097151
 *
 * Integer conversion happens with ceil() as required for converting halo radii to integers
 * where we must round up to the smallest integer not less than x*2^(10 or 21)
 */
template<class KeyType, class T>
HOST_DEVICE_FUN constexpr unsigned toNBitIntCeil(T x)
{
    // spatial resolution in bits per dimension
    constexpr unsigned nBits = maxTreeLevel<KeyType>{};

    // [0,1] to [0,1023] and convert to integer (32-bit) or
    // [0,1] to [0,2097151] and convert to integer (64-bit)
    unsigned result = std::ceil(x * T(1u << nBits));
    return stl::min(result, (1u << nBits) - 1u);
}

/*! @brief add (binary) zeros behind a prefix
 *
 * Allows comparisons, such as number of leading common bits (cpr)
 * of the prefix with SFC codes.
 *
 * @tparam KeyType  32- or 64-bit unsigned integer type
 * @param prefix    the bit pattern
 * @param length    number of bits in the prefix
 * @return          prefix padded out with zeros
 *
 * Examples:
 *  pad(0b011u,  3) -> 0b00011 << 27
 *  pad(0b011ul, 3) -> 0b0011ul << 60
 *
 *  i.e. @p length plus the number of zeros added adds up to 30 for 32-bit integers
 *  or 63 for 64-bit integers, because these are the numbers of usable bits in SFC codes.
 */
template<class KeyType>
constexpr KeyType pad(KeyType prefix, int length)
{
    return prefix << (3 * maxTreeLevel<KeyType>{} - length);
}

/*! @brief compute the maximum range of an octree node at a given subdivision level
 *
 * @tparam KeyType    32- or 64-bit unsigned integer type
 * @param  treeLevel  octree subdivision level
 * @return            the range
 *
 * At treeLevel 0, the range is the entire 30 or 63 bits used in the SFC code.
 * After that, the range decreases by 3 bits for each level.
 *
 */
template<class KeyType>
HOST_DEVICE_FUN constexpr KeyType nodeRange(unsigned treeLevel)
{
    assert(treeLevel <= maxTreeLevel<KeyType>{});
    unsigned shifts = maxTreeLevel<KeyType>{} - treeLevel;

    return KeyType(1ul << (3u * shifts));
}

//! @brief compute ceil(log8(n))
template<class KeyType>
HOST_DEVICE_FUN constexpr unsigned log8ceil(KeyType n)
{
    if (n == 0) { return 0; }

    unsigned lz = countLeadingZeros(n - 1);
    return maxTreeLevel<KeyType>{} - (lz - unusedBits<KeyType>{}) / 3;
}

//! @brief check whether n is a power of 8
template<class KeyType>
HOST_DEVICE_FUN constexpr bool isPowerOf8(KeyType n)
{
    unsigned lz = countLeadingZeros(n - 1) - unusedBits<KeyType>{};
    return lz % 3 == 0 && !(n & (n - 1));
}

/*! @brief calculate common prefix (cpr) of two SFC keys
 *
 * @tparam KeyType  32 or 64 bit unsigned integer
 * @param  key1     first SFC code key
 * @param  key2     second SFC code key
 * @return          number of continuous identical bits, counting from MSB
 *                  minus the 2 unused bits in 32 bit codes or minus the 1 unused bit
 *                  in 64 bit codes.
 */
template<class KeyType>
HOST_DEVICE_FUN constexpr int commonPrefix(KeyType key1, KeyType key2)
{
    return int(countLeadingZeros(key1 ^ key2)) - unusedBits<KeyType>{};
}

/*! @brief return octree subdivision level corresponding to codeRange
 *
 * @tparam KeyType   32- or 64-bit unsigned integer type
 * @param codeRange  input SFC code range
 * @return           octree subdivision level 0-10 (32-bit) or 0-21 (64-bit)
 */
template<class KeyType>
HOST_DEVICE_FUN constexpr unsigned treeLevel(KeyType codeRange)
{
    assert(isPowerOf8(codeRange));
    return (countLeadingZeros(codeRange - 1) - unusedBits<KeyType>{}) / 3;
}

/*! @brief convert a plain SFC key into the placeholder bit format (Warren-Salmon 1993)
 *
 * @tparam KeyType         32- or 64-bit unsigned integer
 * @param code             input SFC key
 * @param prefixLength     number of leading bits which are part of the code
 * @return                 code shifted by trailing zeros and prepended with 1-bit
 *
 * Example: encodePlaceholderBit(06350000000, 9) -> 01635 (in octal)
 */
template<class KeyType>
HOST_DEVICE_FUN constexpr KeyType encodePlaceholderBit(KeyType code, int prefixLength)
{
    int nShifts             = 3 * maxTreeLevel<KeyType>{} - prefixLength;
    KeyType ret             = code >> nShifts;
    KeyType placeHolderMask = KeyType(1) << prefixLength;

    return placeHolderMask | ret;
}

template<class KeyType>
HOST_DEVICE_FUN constexpr KeyType encodePlaceholderBit2K(KeyType k1, KeyType k2)
{
    //! prefixLength is 3 * treeLevel(endKey - startKey)
    unsigned prefixLength = countLeadingZeros(k2 - k1 - 1) - unusedBits<KeyType>{};
    return encodePlaceholderBit(k1, prefixLength);
}

//! @brief returns the number of key-bits in the input @p code
template<class KeyType>
HOST_DEVICE_FUN constexpr unsigned decodePrefixLength(KeyType code)
{
    return 8 * sizeof(KeyType) - 1 - countLeadingZeros(code);
}

/*! @brief decode an SFC key in Warren-Salmon placeholder bit format
 *
 * @tparam KeyType   32- or 64-bit unsigned integer
 * @param code       input SFC key with 1-bit prepended
 * @return           SFC-key without 1-bit and shifted to most significant bit
 *
 * Inverts encodePlaceholderBit.
 */
template<class KeyType>
HOST_DEVICE_FUN constexpr KeyType decodePlaceholderBit(KeyType code)
{
    int prefixLength        = decodePrefixLength(code);
    KeyType placeHolderMask = KeyType(1) << prefixLength;
    KeyType ret             = code ^ placeHolderMask;

    return ret << (3 * maxTreeLevel<KeyType>{} - prefixLength);
}

/*! @brief decode an SFC key in Warren-Salmon placeholder bit format
 *
 * @tparam KeyType   32- or 64-bit unsigned integer
 * @param code       input SFC key with 1-bit prepended
 * @return           SFC-key without 1-bit and shifted to most significant bit
 *
 * Inverts encodePlaceholderBit.
 */
template<class KeyType>
HOST_DEVICE_FUN constexpr util::tuple<KeyType, KeyType> decodePlaceholderBit2K(KeyType code)
{
    int prefixLength        = decodePrefixLength(code);
    KeyType placeHolderMask = KeyType(1) << prefixLength;
    KeyType ret             = code ^ placeHolderMask;

    int nShifts = 3 * maxTreeLevel<KeyType>{} - prefixLength;
    auto k1     = ret << nShifts;
    return {k1, k1 + (KeyType(1) << nShifts)};
}

//! @brief locate with @p nodeKey given in Warren-Salmon placeholder-bit format
template<class KeyType>
HOST_DEVICE_FUN TreeNodeIndex locateNode(KeyType nodeKey, const KeyType* prefixes, const TreeNodeIndex* levelRange)
{
    TreeNodeIndex numNodes = levelRange[maxTreeLevel<KeyType>{} + 1];
    unsigned level         = decodePrefixLength(nodeKey) / 3;
    auto it                = stl::lower_bound(prefixes + levelRange[level], prefixes + levelRange[level + 1], nodeKey);
    if (it != prefixes + numNodes && *it == nodeKey) { return it - prefixes; }
    else { return numNodes; }
}

/*! @brief finds the index of the node with SFC key range [startKey:endKey]
 *
 * @param startKey   lower SFC key
 * @param endKey     upper SFC key
 * @return           The index i of the node that satisfies codeStart(i) == startKey
 *                   and codeEnd(i) == endKey, or numTreeNodes() if no such node exists.
 */
template<class KeyType>
HOST_DEVICE_FUN TreeNodeIndex
locateNode(KeyType startKey, KeyType endKey, const KeyType* prefixes, const TreeNodeIndex* levelRange)
{
    //! prefixLength is 3 * treeLevel(endKey - startKey)
    unsigned prefixLength = countLeadingZeros(endKey - startKey - 1) - unusedBits<KeyType>{};
    return locateNode(encodePlaceholderBit(startKey, prefixLength), prefixes, levelRange);
}

//! @brief Mask key to set special status. Does not support WS-prefix keys.
template<class KeyType>
KeyType maskKey(KeyType key)
{
    if (key == 0 || key == nodeRange<KeyType>(0)) { return key; }
    return key | nodeRange<KeyType>(0);
}

//! @brief Inverse of maskKey
template<class KeyType>
KeyType unmaskKey(KeyType key)
{
    if (key == nodeRange<KeyType>(0)) { return key; }
    return key & (nodeRange<KeyType>(0) - 1);
}

template<class KeyType>
bool isMasked(KeyType key)
{
    return key > nodeRange<KeyType>(0);
}

/*! @brief extract the n-th octal digit from an SFC key, starting from the most significant
 *
 * @tparam KeyType   32- or 64-bit unsigned integer type
 * @param code       Input SFC key code
 * @param position   Which digit place to extract. Return values will be meaningful for
 *                   @p position in [1:11] for 32-bit keys and in [1:22] for 64-bit keys and
 *                   will be zero otherwise, but a value of 0 for @p position can also be specified
 *                   to detect whether the 31st or 63rd bit for the last cornerstone is non-zero.
 *                   (The last cornerstone has a value of nodeRange<KeyType>(0) = 2^31 or 2^63)
 * @return           The value of the digit at place @p position
 *
 * The position argument correspondence to octal digit places has been chosen such that
 * octalDigit(code, pos) returns the octant at octree division level pos.
 */
template<class KeyType>
HOST_DEVICE_FUN constexpr unsigned octalDigit(KeyType code, unsigned position)
{
    return (code >> (3u * (maxTreeLevel<KeyType>{} - position))) & 7u;
}

//! @brief return true if a is an ancestor of b or if a is a sibling of an ancestor of b
template<class KeyType>
HOST_DEVICE_FUN constexpr bool isAncestor(KeyType a, KeyType b)
{
    int alen = decodePrefixLength(a);
    int blen = decodePrefixLength(b);

    a <<= stl::max(0, blen - alen);
    auto commonBits = countLeadingZeros(a ^ b);

    return commonBits >= 1 + countLeadingZeros(b) + stl::max(0, alen - 3);
}

//! @brief return the offset octal digit weight for binary tree <-> octree index correspondence
HOST_DEVICE_FUN constexpr int digitWeight(int digit)
{
    int fourGeqMask = -int(digit >= 4);
    return ((7 - digit) & fourGeqMask) - (digit & ~fourGeqMask);
}

//! @brief cut down the input SFC code to the start code of the enclosing box at <treeLevel>
template<class KeyType>
HOST_DEVICE_FUN constexpr KeyType enclosingBoxCode(KeyType key, unsigned treeLevel)
{
    KeyType mask = KeyType(nodeRange<KeyType>(treeLevel) - 1);

    return KeyType(key & ~mask);
}

/*! @brief compute an enclosing envelope corresponding to the smallest possible
 *         octree node for two input SFC codes
 *
 * @tparam    KeyType    32- or 64-bit unsigned integer type
 * @param[in] firstKey   first SFC key
 * @param[in] secondKey  second SFC key
 * @return               two SFC keys that delineate the start and end of
 *                       the smallest octree node that contains both input keys
 */
template<class KeyType>
HOST_DEVICE_FUN util::tuple<KeyType, KeyType> smallestCommonBox(KeyType firstKey, KeyType secondKey)
{
    unsigned commonLevel = commonPrefix(firstKey, secondKey) / 3;
    KeyType nodeStart    = enclosingBoxCode(firstKey, commonLevel);

    return {nodeStart, nodeStart + nodeRange<KeyType>(commonLevel)};
}

//! @brief zero all but the highest nBits in a SFC code
template<class KeyType>
HOST_DEVICE_FUN constexpr KeyType zeroLowBits(KeyType code, unsigned nBits)
{
    unsigned nLowerBits = 3 * maxTreeLevel<KeyType>{} - nBits;
    KeyType mask        = (KeyType(1) << nLowerBits) - 1;

    return code & ~mask;
}

/*! @brief return position of last non-zero octal digit place in x
 *
 * @tparam KeyType   32- or 64-bit unsigned integer
 * @param  x         an integer
 * @return           position of the last non-zero octal digit place, starting from
 *                   1 (left-most digit place) to 10 or 21 (64-bit), the right-most
 *                   digit place. Returns 10 or 21 (64-bit) if x is zero.
 */
template<class KeyType>
HOST_DEVICE_FUN constexpr int lastNzPlace(KeyType x)
{
    if (x)
        return maxTreeLevel<KeyType>{} - countTrailingZeros(x) / 3;
    else
        return maxTreeLevel<KeyType>{};
}

//! @brief compute the placeholder-bit prefix for the biggest possible node that starts at @p a
template<class KeyType>
HOST_DEVICE_FUN constexpr KeyType makePrefix(KeyType a)
{
    if (a == 0) { return 1; }

    int level = lastNzPlace(a);
    return encodePlaceholderBit(a, 3 * level);
}

/*! @brief return the power of 8 for the octal place at position @p pos
 *
 * @tparam KeyType    32- or 64-bit unsigned integer
 * @param  pos  Position counting from left, starting from 1. Maximum value 10 or 21 (64-bit)
 * @return      the power of 8 associated with the indicated octal place
 */
template<class KeyType>
HOST_DEVICE_FUN constexpr KeyType octalPower(int pos)
{
    return (KeyType(1) << 3 * (maxTreeLevel<KeyType>{} - pos));
}

/*! @brief generate SFC codes to cover the range [a:b] with a valid cornerstone sub-octree
 *
 * @tparam     KeyType 32- or 64-bit unsigned integer
 * @tparam     Store   either std::nullptr_t or KeyType*
 * @param[in]  a       first SFC code
 * @param[in]  b       second SFC code, b > a
 * @param[out] output  output SFC codes, includes a, excludes b
 * @return             number of values in output
 *
 *                      | a_last_nz_pos (10)
 * Example:       a: 0001
 *                b: 0742
 *  ab_first_diff_pos ^ ^ b_last_nz_pos
 *       (8)                   (10)
 *
 *  output: 1 2 3 4 5 6 7 10 20 30 40 50 60 70 100 200 300 400 500 600 700 710 720 730 740 741
 *
 *  Variables suffixed with "_pos" refer to an octal digit place. The value of 1 is
 *  the position of the left-most digit, and 10 (or 21 for 64-bit) refers to the right-most digit place.
 *  This convention is chosen such that the positional value coincides with the corresponding octree
 *  subdivision level.
 */
template<class KeyType, class Store>
HOST_DEVICE_FUN std::enable_if_t<std::is_same_v<Store, std::nullptr_t> || std::is_same_v<Store, KeyType*>, int>
spanSfcRange(KeyType a, KeyType b, [[maybe_unused]] Store output)
{
    int numValues = 0;
    // position of first differing octal digit place
    int ab_first_diff_pos = (countLeadingZeros(a ^ b) + 3 - unusedBits<KeyType>{}) / 3;

    // last non-zero octal digit place position in a and b
    int a_last_nz_pos = lastNzPlace(a);
    int b_last_nz_pos = lastNzPlace(b);

    // add SFC codes, increasing power of 8 in each iteration
    for (int pos = a_last_nz_pos; pos > ab_first_diff_pos; --pos)
    {
        int numDigits = (8 - octalDigit(a, pos)) % 8;
        numValues += numDigits;
        while (numDigits--)
        {
            if constexpr (!std::is_same_v<Store, std::nullptr_t>) { *output++ = a; }
            a += octalPower<KeyType>(pos);
        }
    }

    // add SFC codes, decreasing power of 8 in each iteration
    for (int pos = ab_first_diff_pos; pos <= b_last_nz_pos; ++pos)
    {
        // Note: octalDigit(a, pos) is guaranteed zero for pos > ab_first_diff_pos
        int numDigits = octalDigit(b, pos) - octalDigit(a, pos);
        numValues += numDigits;
        while (numDigits--)
        {
            if constexpr (!std::is_same_v<Store, std::nullptr_t>) { *output++ = a; }
            a += octalPower<KeyType>(pos);
        }
    }

    return numValues;
}

//! @brief overload to skip storage and just compute number of values, see spanSfcRange(KeyType a, KeyType b, KeyType*
//! output) above
template<class KeyType>
HOST_DEVICE_FUN int spanSfcRange(KeyType a, KeyType b)
{
    return spanSfcRange<KeyType, std::nullptr_t>(a, b, nullptr);
}

} // namespace cstone
