/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @brief @file parallel binary radix tree construction implementation
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * Algorithm published in https://dl.acm.org/doi/10.5555/2383795.2383801
 * and further illustrated at
 * https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
 *
 * Most SFC-based functionality for SPH like domain decomposition, global octree build
 * and neighbor search does not need tree-traversal and explicit construction
 * of tree-internal nodes.
 *
 * The halo search problem is an exception. Finding all nodes in the global octree
 * that overlap with a given node that is enlarged by the search radius in all
 * dimensions is essentially the same as collision detection between objects
 * in 3D graphics, where bounding volume hierarchies are constructed from binary radix trees.
 *
 * The reason why the halo search problem differs from the other SFC-based algorithms
 * for neighbor search and domain decomposition is that an octree node that is enlarged
 * by an arbitrary distance in each dimension cannot necessarily be composed from octree
 * nodes of similar size as the query node and therefore also not as a range of SFC codes,
 * or the sum of a small number of SFC code ranges. This is especially true at the boundaries
 * between octree nodes of different division level, i.e. when large nodes are next to
 * very small nodes. For this reason, nodes enlarged by a halo radius are best represented
 * by x,y,z coordinate ranges.
 */

#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/sfc/common.hpp"

#include "definitions.h"

namespace cstone
{

//! @brief checks whether a binary tree index corresponds to a leaf index
HOST_DEVICE_FUN constexpr bool isLeafIndex(TreeNodeIndex nodeIndex) { return nodeIndex < 0; }

//! @brief convert a leaf index to the storage format
HOST_DEVICE_FUN constexpr TreeNodeIndex storeLeafIndex(TreeNodeIndex index)
{
    // -2^31 or -2^63
    constexpr auto offset = TreeNodeIndex(-(1ul << (8 * sizeof(TreeNodeIndex) - 1)));
    return index + offset;
}

//! @brief restore a leaf index from the storage format
HOST_DEVICE_FUN constexpr TreeNodeIndex loadLeafIndex(TreeNodeIndex index)
{
    constexpr auto offset = TreeNodeIndex(-(1ul << (8 * sizeof(TreeNodeIndex) - 1)));
    return index - offset;
}

/*! @brief binary radix tree node
 *
 * @tparam I 32- or 64 bit unsigned integer
 */
template<class I>
struct BinaryNode
{
    enum ChildType
    {
        left  = 0,
        right = 1
    };

    /*! @brief indices to the left and right children nodes
     *
     * The left child adds a 0-bit to the prefix, while
     * the right child adds a 1-bit to the prefix.
     *
     * Leaf indices are stored as negative values to differentiate them
     * from indices of internal nodes. Use isLeafIndex for querying the leaf property
     * and btreeLoad/StoreLeaf to convert to the actual positive leaf index into
     * the SFC code array used to construct the binary tree (e.g. the cornerstone tree).
     */
    TreeNodeIndex child[2];

    //! @brief the SFC key code with placeholder bit (Warren-Salmon 1993)
    I prefix;
};

/*! @brief find position of first differing bit
 *
 * @tparam I                     32 or 64 bit unsigned integer
 * @param[in] sortedSfcCodes     sorted SFC codes
 * @param[in] first              first range index
 * @param[in] last               last rang index
 * @return                       position of morton
 */
template<class I>
HOST_DEVICE_FUN TreeNodeIndex findSplit(const I* sortedSfcCodes, TreeNodeIndex first, TreeNodeIndex last)
{
    // Identical Morton codes => split the range in the middle.
    I firstCode = sortedSfcCodes[first];
    I lastCode  = sortedSfcCodes[last];

    if (firstCode == lastCode) return (first + last) >> 1;

    // Calculate the number of highest bits that are the same for all objects
    int cpr = commonPrefix(firstCode, lastCode);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest Morton code that
    // shares more than commonPrefix bits with the first one.
    TreeNodeIndex split = first; // initial guess
    TreeNodeIndex step  = last - first;

    do
    {
        step                   = (step + 1) / 2; // exponential decrease
        TreeNodeIndex newSplit = split + step;   // proposed new position

        if (newSplit < last)
        {
            I splitCode               = sortedSfcCodes[newSplit];
            TreeNodeIndex splitPrefix = commonPrefix(firstCode, splitCode);
            if (splitPrefix > cpr) split = newSplit; // accept proposal
        }
    } while (step > 1);

    return split;
}

/*! @brief construct the internal binary tree node with index idx
 *
 * @tparam I                  32- or 64-bit unsigned integer
 * @param[in]  codes          sorted Morton code sequence without duplicates
 * @param[in]  nCodes         number of elements in @p codes
 * @param[out] internalNodes  output internal binary radix tree, size is nCodes - 1
 * @param[in]  firstIndex     element of @p internalNodes to construct,
 *                            permissible range is 0 <= firstIndex < nCodes-1
 */
template<class I>
HOST_DEVICE_FUN void
constructInternalNode(const I* codes, TreeNodeIndex nCodes, BinaryNode<I>* internalNodes, TreeNodeIndex firstIndex)
{
    BinaryNode<I> outputNode;

    int d               = 1;
    int minPrefixLength = -1;

    if (firstIndex > 0)
    {
        d               = (commonPrefix(codes[firstIndex], codes[firstIndex + 1]) >
             commonPrefix(codes[firstIndex], codes[firstIndex - 1]))
                              ? 1
                              : -1;
        minPrefixLength = commonPrefix(codes[firstIndex], codes[firstIndex - d]);
    }

    // determine searchRange, the maximum distance of secondIndex from firstIndex
    TreeNodeIndex searchRange = 2;
    TreeNodeIndex secondIndex = firstIndex + searchRange * d;
    while (0 <= secondIndex && secondIndex < nCodes &&
           commonPrefix(codes[firstIndex], codes[secondIndex]) > minPrefixLength)
    {
        searchRange *= 2;
        secondIndex = firstIndex + searchRange * d;
    }

    // start binary search with known searchRange
    secondIndex = firstIndex;
    do
    {
        searchRange          = (searchRange + 1) / 2;
        TreeNodeIndex newJdx = secondIndex + searchRange * d;
        if (0 <= newJdx && newJdx < nCodes && commonPrefix(codes[firstIndex], codes[newJdx]) > minPrefixLength)
        {
            secondIndex = newJdx;
        }
    } while (searchRange > 1);

    int prefixLength  = commonPrefix(codes[firstIndex], codes[secondIndex]);
    I prefix          = zeroLowBits(codes[firstIndex], prefixLength);
    outputNode.prefix = encodePlaceholderBit(prefix, prefixLength);

    // find position of highest differing bit between [firstIndex, secondIndex]
    TreeNodeIndex gamma = findSplit(codes, stl::min(secondIndex, firstIndex), stl::max(secondIndex, firstIndex));

    // establish child relationships
    if (stl::min(secondIndex, firstIndex) == gamma)
    {
        // left child is a leaf
        outputNode.child[BinaryNode<I>::left] = storeLeafIndex(gamma);
    }
    else
    {
        // left child is an internal binary node
        outputNode.child[BinaryNode<I>::left] = gamma;
    }

    if (stl::max(secondIndex, firstIndex) == gamma + 1)
    {
        // right child is a leaf
        outputNode.child[BinaryNode<I>::right] = storeLeafIndex(gamma + 1);
    }
    else
    {
        // right child is an internal binary node
        outputNode.child[BinaryNode<I>::right] = gamma + 1;
    }

    internalNodes[firstIndex] = outputNode;
}

/*! @brief create a binary radix tree from a cornerstone octree
 *
 * @tparam     KeyType        32- or 64-bit unsigned integer
 * @param[in]  tree           Sorted Morton codes representing the leaves of the (global) octree
 *                            or the locations of objects in 3D.
 *                            Cornerstone invariants are not a requirement for this function,
 *                            only that the codes be sorted and not contain any duplicates.
 * @param[in]  nNodes         nNodes == length(tree) - 1
 *                            If @p tree is in cornerstone format, nNodes is the number of leaf nodes.
 * @param[out] binaryTree     output binary tree, length == @p nNodes
 * @return                    the internal part of the input tree constructed as binary nodes
 *
 * Note that if the input @p tree is a cornerstone octree, the root node with index
 * 0 in the returned binary tree only maps binary nodes 0 <= ... < tree.size() -1.
 * Due to the last element of tree being the maximum Morton code 2^(30 or 61),
 * the last node/element of the returned binary tree will be set up as a useless
 * second root node that is not reachable from the root node with index 0.
 * So if @p tree is a cornerstone octree with an array size of N, we can say that
 *      - @p tree has N-1 octree leaf nodes
 *      - the output is a binary tree of array size N-1 with 0...N-2 as usable elements
 *
 * One could of course prevent the generation of the last binary node with index N-1,
 * but that would result in loss of generality for arbitrary sorted Morton code sequences
 * without duplicates.
 */
template<class KeyType>
void createBinaryTree(const KeyType* tree, TreeNodeIndex nNodes, BinaryNode<KeyType>* binaryTree)
{
#pragma omp parallel for
    for (TreeNodeIndex idx = 0; idx < nNodes; ++idx)
    {
        constructInternalNode(tree, nNodes + 1, binaryTree, idx);
    }
}

} // namespace cstone
