/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

#pragma once

#include <cassert>
#include <type_traits>
#include "cstone/cuda/annotation.hpp"

#include "cstone/primitives/stl.hpp"
#include "cstone/util/array.hpp"

namespace cstone
{

/*! @brief
 * Controls the node index type, has to be signed. Change to 64-bit if more than 2 billion tree nodes are required.
 */
using TreeNodeIndex = int;
//! @brief index type of local particle arrays
using LocalIndex = unsigned;

template<class KeyType>
struct unusedBits
{
};

//! @brief number of unused leading zeros in a 32-bit SFC code
template<>
struct unusedBits<unsigned> : stl::integral_constant<unsigned, 2>
{
};

//! @brief number of unused leading zeros in a 64-bit SFC code
template<>
struct unusedBits<unsigned long long> : stl::integral_constant<unsigned, 1>
{
};
template<>
struct unusedBits<unsigned long> : stl::integral_constant<unsigned, 1>
{
};

template<class KeyType>
struct maxTreeLevel
{
};

template<>
struct maxTreeLevel<unsigned> : stl::integral_constant<unsigned, 10>
{
};

template<>
struct maxTreeLevel<unsigned long long> : stl::integral_constant<unsigned, 21>
{
};
template<>
struct maxTreeLevel<unsigned long> : stl::integral_constant<unsigned, 21>
{
};

//! @brief A special key value that cannot result from valid coordinates. Used to flag particles for removal.
template<class KeyType>
struct removeKey
{
    static constexpr KeyType value = KeyType(1ul << (3 * maxTreeLevel<KeyType>{}));
    HOST_DEVICE_FUN constexpr operator KeyType() const noexcept { return value; }
};

//! @brief maximum integer coordinate
template<class KeyType>
struct maxCoord : stl::integral_constant<unsigned, (1u << maxTreeLevel<KeyType>{})>
{
};

template<class T>
using Vec3 = util::array<T, 3>;

template<class T>
using Vec4 = util::array<T, 4>;

enum class P2pTags : int
{
    focusTransfer    = 1000,
    focusTreelets    = 2000,
    focusPeerCounts  = 3000,
    focusPeerCenters = 4000,
    haloRequestKeys  = 5000,
    domainExchange   = 6000,
    haloExchange     = 7000
};

/*! @brief returns the number of nodes in a tree
 *
 * @tparam    Vector  a vector-like container that has a .size() member
 * @param[in] tree    input tree
 * @return            the number of nodes
 *
 * This makes it explicit that a vector of n Morton codes
 * corresponds to a tree with n-1 nodes.
 */
template<class Vector>
std::size_t nNodes(const Vector& tree)
{
    assert(tree.size());
    return tree.size() - 1;
}

} // namespace cstone
