/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief utility functions for cornerstone octrees
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * The functionality in this file is primarily used to test the cornerstone
 * octree implementation, but might be useful in production code as well.
 *
 */

#pragma once

#include <array>

#include "cstone/sfc/common.hpp"
#include "cstone/tree/csarray.hpp"

namespace cstone
{

/*! @brief check whether the cornerstone octree format invariants are fulfilled
 *
 * @tparam I           32- or 64-bit unsigned integer type
 * @param tree         octree nodes given as Morton codes of length @a nNodes+1
 * @param nNodes       number of nodes
 * @return             true if invariants ar satisfied, false otherwise
 *
 * The invariants are:
 *      - tree contains code 0 and the maximum code 2^30 or 2^61
 *      - tree is sorted
 *      - difference between consecutive elements must be a power of 8
 */
template<class I>
bool checkOctreeInvariants(const I* tree, int nNodes)
{
    // the root node delineated by code 0 and nodeRange<KeyType>(0)
    // must be part of the tree
    if (nNodes < 1) return false;
    if (tree[0] != 0 || tree[nNodes] != nodeRange<I>(0)) return false;

    for (int i = 0; i < nNodes; ++i)
    {
        if (i + 1 < nNodes && tree[i] >= tree[i + 1]) return false;

        I range = tree[i + 1] - tree[i];

        if (range == 0) return false;

        if (!isPowerOf8(range)) return false;
    }

    return true;
}

//! @brief returns an octree with just the root node
template<class I>
std::vector<I> makeRootNodeTree()
{
    std::vector<I> tree;

    tree.push_back(0);
    tree.push_back(nodeRange<I>(0));

    return tree;
}

//! @brief returns a uniform grid of SFC codes
template<class I>
std::vector<I> makeNLevelGrid(int level)
{
    unsigned ticksPerDim = 1u << level;
    I nNodes             = ticksPerDim * ticksPerDim * ticksPerDim;
    I tickRange          = nodeRange<I>(level);

    std::vector<I> codes;
    codes.reserve(nNodes + 1);

    for (unsigned i = 0; i < nNodes; ++i)
    {
        codes.push_back(i * tickRange);
    }

    return codes;
}

//! @brief returns a uniform cornerstone octree with 8^ceil(log8(nBuckets)) nodes
template<class I>
std::vector<I> makeUniformNLevelTree(std::size_t nParticles, int bucketSize)
{
    unsigned level      = log8ceil(I(nParticles / std::size_t(bucketSize)));
    std::vector<I> tree = makeNLevelGrid<I>(level);

    tree.push_back(nodeRange<I>(0));

    return tree;
}

/*! @brief transfer a series of hierarchical octree indices into a morton code
 *
 * @tparam I       32- or 64-bit unsigned integer
 * @param indices  indices[0] contains the octree index 0-7 for the top-level,
 *                 indices[1] refers to the first subdivision, etc
 *                 a 32-bit integer can resolve up to 10 layers, while
 *                 a 64-bit integer can resolve 21 layers
 *
 *                 Note: all indices must be in the range [0-7]!
 *
 * @return         the morton code
 */
template<class I>
inline I codeFromIndices(std::array<unsigned char, maxTreeLevel<uint64_t>{}> indices)
{
    constexpr unsigned nLevels = (sizeof(I) * 8) / 3;

    I ret = 0;
    for (unsigned idx = 0; idx < nLevels; ++idx)
    {
        assert(indices[idx] < 8);
        unsigned treeLevel = nLevels - idx - 1;
        ret += I(indices[idx]) << (3 * treeLevel);
    }

    return ret;
}

//! @brief generate example cornerstone octrees for testing
template<class I>
class OctreeMaker
{
public:
    OctreeMaker()
        : tree(makeRootNodeTree<I>())
    {
    }

    /*! @brief introduce all 8 children of the node specified as argument
     *
     * @param idx    node definition given as a series of indices in [0-7],
     *               as specified by the function codeFromIndices.
     * @param level  number of indices in idx that belong to the node to be divided
     * @return       the object itself to allow chaining of divide calls()
     *
     * This function adds the Morton codes codeFromIndices({args..., i}) for i = 1...7
     * to the tree which corresponds to dividing the existing node codeFromIndices({args...});
     */
    OctreeMaker& divide(std::array<int, maxTreeLevel<uint64_t>{}> idx, std::size_t level)
    {
        std::array<unsigned char, maxTreeLevel<uint64_t>{}> indices{};
        for (std::size_t i = 0; i < idx.size(); ++i)
            indices[i] = static_cast<unsigned char>(idx[i]);

        assert(std::find(begin(tree), end(tree), codeFromIndices<I>(indices)) != end(tree) &&
               "node to be divided not present in tree");

        indices[level] = 1;
        assert(std::find(begin(tree), end(tree), codeFromIndices<I>(indices)) == end(tree) &&
               "children of node to be divided already present in tree");

        for (std::size_t sibling = 1; sibling < 8; ++sibling)
        {
            indices[level] = sibling;
            tree.push_back(codeFromIndices<I>(indices));
        }

        return *this;
    }

    /*! @brief convenience alias for the other divide
     *
     * Gets rid of the explicit level argument which is not needed if the number
     * of levels is known at compile time.
     */
    template<class... Args>
    OctreeMaker& divide(Args... args)
    {
        return divide({args...}, sizeof...(Args));
    }

    //! @brief return the finished tree, fulfilling the necessary invariants
    std::vector<I> makeTree()
    {
        std::sort(begin(tree), end(tree));
        return tree;
    }

private:
    std::vector<I> tree;
};

} // namespace cstone
