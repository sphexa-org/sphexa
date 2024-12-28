/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Generation of local and global octrees in cornerstone format
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * In the cornerstone format, the octree is stored as sequence of SFC codes
 * fulfilling three invariants. Each code in the sequence both signifies the
 * start SFC code of an octree leaf node and serves as an upper SFC code bound
 * for the previous node.
 *
 * The invariants of the cornerstone format are:
 *      - code sequence contains code 0 and the maximum code 2^30 or 2^61
 *      - code sequence is sorted by ascending code value
 *      - difference between consecutive elements must be a power of 8
 *
 * The consequences of these invariants are:
 *      - the entire space is covered, i.e. there are no holes in the tree
 *      - only leaf nodes are stored
 *      - for each leaf node, all its siblings (nodes at the same subdivision level with
 *        the same parent) are present in the SFC code sequence
 *      - each node with index i is defined by its lowest possible SFC code at position
 *        i in the vector and the highest possible (excluding) SFC code at position i+1
 *        in the vector
 *      - a vector of length N represents N-1 leaf nodes
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <vector>
#include <tuple>

#include "cstone/sfc/common.hpp"
#include "cstone/util/tuple.hpp"

#include "definitions.h"

namespace cstone
{

/*! @brief Fixed number of children per node
 *
 * This is beneficial for performance as long as the bucket-size, which is the minimum particle count of an internal
 * node and the maximum count of a leaf, remains well above 8 and becomes very disadvantageous when resolving
 * every particle in its own leaf node.
 */
constexpr inline TreeNodeIndex eightSiblings = 8;

//! @brief return first node that starts at or below (contains) key
template<class KeyType>
HOST_DEVICE_FUN inline TreeNodeIndex findNodeBelow(const KeyType* tree, TreeNodeIndex numNodes, KeyType key)
{
    return stl::upper_bound(tree, tree + numNodes, key) - tree - 1;
}

//! @brief return first node that starts at or above key
template<class KeyType>
HOST_DEVICE_FUN inline TreeNodeIndex findNodeAbove(const KeyType* tree, TreeNodeIndex numNodes, KeyType key)
{
    return stl::lower_bound(tree, tree + numNodes, key) - tree;
}

//! @brief count particles in one tree node
template<class KeyType>
HOST_DEVICE_FUN unsigned calculateNodeCount(
    KeyType nodeStart, KeyType nodeEnd, const KeyType* codesStart, const KeyType* codesEnd, size_t maxCount)
{
    // count particles in range
    auto rangeStart = stl::lower_bound(codesStart, codesEnd, nodeStart);
    auto rangeEnd   = stl::lower_bound(codesStart, codesEnd, nodeEnd);
    size_t count    = rangeEnd - rangeStart;

    return stl::min(count, maxCount);
}

/*! @brief Narrow search bound for @p targetCode in an array of sorted particle SFC codes
 *
 * @tparam KeyType    32- or 64-bit unsigned integer type
 * @param firstIdx    first (of two) search boundaries, may be out of bounds
 *                    (the guess for the location of @p targetCode in [codesStart:codesEnd]
 * @param targetCode  code to look for in [codesStart:codesEnd]
 * @param codesStart  particle SFC code array start
 * @param codesEnd    particle SFC code array end
 * @return            a sub-range of [codesStart:codesEnd] that fulfills
 *                    lower_bound(codesStart, codesEnd, targetCode) == lower_bound(ret[0], ret[0], targetCode)
 */
template<class KeyType>
HOST_DEVICE_FUN util::array<const KeyType*, 2> findSearchBounds(std::make_signed_t<KeyType> firstIdx,
                                                                KeyType targetCode,
                                                                const KeyType* codesStart,
                                                                const KeyType* codesEnd)
{
    using SI  = std::make_signed_t<KeyType>;
    SI nCodes = codesEnd - codesStart;
    if (nCodes == 0) { return {codesStart, codesStart}; }

    // firstIdx must be an accessible index
    firstIdx = stl::max(firstIdx, SI(0));
    firstIdx = stl::min(nCodes - 1, firstIdx);

    // the last node in 64-bit is 2^63, which can't be represented as a negative number
    if (targetCode >= nodeRange<KeyType>(0)) { return {stl::lower_bound(codesStart, codesEnd, targetCode), codesEnd}; }

    // d = 1 : search towards codesEnd, d = -1 : search towards codesStart
    SI d = (codesStart[firstIdx] < targetCode) ? 1 : -1;

    SI targetCodeTimesD = targetCode * d;

    // determine search bound
    SI searchRange = 1;
    SI secondIndex = firstIdx + d;
    while (0 <= secondIndex && secondIndex < nCodes && d * SI(codesStart[secondIndex]) <= targetCodeTimesD)
    {
        searchRange *= 2;
        secondIndex = firstIdx + searchRange * d;
    }
    secondIndex = stl::max(SI(0), secondIndex);
    secondIndex = stl::min(nCodes, secondIndex);

    return {codesStart + stl::min(firstIdx, secondIndex), codesStart + stl::max(firstIdx, secondIndex)};
}

/*! @brief calculate node counts with a guess to accelerate the binary search
 *
 * @tparam KeyType          32- or 64-bit unsigned integer type
 * @param nodeIdx           the index of the node in @p tree to compute
 * @param tree              cornerstone octree
 * @param firstGuess        guess location of @p tree[nodeIdx] in [codesStart:codesEnd]
 * @param secondGuess       guess location of @p tree[nodeIdx+1] in [codesStart:codesEnd]
 * @param codesStart        particle SFC code array start
 * @param codesEnd          particle SFC code array end
 * @param maxCount          maximum particle count to report per node
 * @return                  the number of particles in the node at @p nodeIdx or maxCount,
 *                          whichever is smaller
 */
template<class KeyType>
HOST_DEVICE_FUN unsigned updateNodeCount(TreeNodeIndex nodeIdx,
                                         const KeyType* tree,
                                         std::make_signed_t<KeyType> firstGuess,
                                         std::make_signed_t<KeyType> secondGuess,
                                         const KeyType* codesStart,
                                         const KeyType* codesEnd,
                                         unsigned maxCount)
{
    KeyType nodeStart = tree[nodeIdx];
    KeyType nodeEnd   = tree[nodeIdx + 1];

    auto searchBounds = findSearchBounds(firstGuess, nodeStart, codesStart, codesEnd);
    auto rangeStart   = stl::lower_bound(searchBounds[0], searchBounds[1], nodeStart);

    searchBounds  = findSearchBounds(secondGuess, nodeEnd, codesStart, codesEnd);
    auto rangeEnd = stl::lower_bound(searchBounds[0], searchBounds[1], nodeEnd);

    unsigned count = rangeEnd - rangeStart;
    return stl::min(count, maxCount);
}

/*! @brief count number of particles in each octree node
 *
 * @tparam       KeyType      32- or 64-bit unsigned integer type
 * @param[in]    tree         octree nodes given as SFC codes of length @a nNodes+1
 *                            needs to satisfy the octree invariants
 * @param[inout] counts       output particle counts per node, length = @a nNodes
 * @param[in]    nNodes       number of nodes in tree
 * @param[in]    keys         sorted particle SFC keys
 * @param[in]    maxCount     maximum particle count per node to store, this is used
 *                            to prevent overflow in MPI_Allreduce
 */
template<class KeyType>
void computeNodeCounts(const KeyType* tree,
                       unsigned* counts,
                       TreeNodeIndex nNodes,
                       std::span<const KeyType> keys,
                       unsigned maxCount,
                       bool useCountsAsGuess = false)
{
    TreeNodeIndex firstNode = 0;
    TreeNodeIndex lastNode  = nNodes;
    if (not keys.empty())
    {
        firstNode = std::upper_bound(tree, tree + nNodes, keys.front()) - tree - 1;
        lastNode  = std::upper_bound(tree, tree + nNodes, keys.back()) - tree;
        assert(firstNode <= lastNode && "Are your particle codes sorted?");
    }
    else
    {
        firstNode = nNodes;
        lastNode  = nNodes;
    }

#pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < firstNode; ++i)
        counts[i] = 0;

#pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = lastNode; i < nNodes; ++i)
        counts[i] = 0;

    TreeNodeIndex nNonZeroNodes  = lastNode - firstNode;
    const KeyType* populatedTree = tree + firstNode;

    if (useCountsAsGuess)
    {
        std::exclusive_scan(counts + firstNode, counts + firstNode + nNonZeroNodes, counts + firstNode, LocalIndex{0});
#pragma omp parallel for schedule(static)
        for (TreeNodeIndex i = 0; i < nNonZeroNodes; ++i)
        {
            unsigned firstGuess   = counts[i + firstNode];
            unsigned secondGuess  = counts[std::min(i + firstNode + 1, nNodes - 1)];
            counts[i + firstNode] = updateNodeCount(i, populatedTree, firstGuess, secondGuess, keys.data(),
                                                    keys.data() + keys.size(), maxCount);
        }
    }
    else
    {
#pragma omp parallel for schedule(static)
        for (TreeNodeIndex i = 0; i < nNonZeroNodes; ++i)
        {
            counts[i + firstNode] = calculateNodeCount(populatedTree[i], populatedTree[i + 1], keys.data(),
                                                       keys.data() + keys.size(), maxCount);
        }
    }
}

/*! @brief return the sibling index and level of the specified csTree node
 *
 * @tparam KeyType   32- or 64-bit unsigned integer
 * @param  csTree    cornerstone octree, length N
 * @param  nodeIdx   node index in [0:N] of @p csTree to compute sibling index
 * @return           in first pair element: index in [0:8] if all 8 siblings of the specified
 *                   node are next to each other and at the same division level.
 *                   -1 otherwise, i.e. if not all the 8 siblings exist in @p csTree
 *                   at the same division level
 *                   in second pair element: tree level of node at @p nodeIdx
 *
 * Sibling nodes are group of 8 leaf nodes that have the same parent node.
 */
template<class KeyType>
inline HOST_DEVICE_FUN util::tuple<int, unsigned> siblingAndLevel(const KeyType* csTree, TreeNodeIndex nodeIdx)
{
    KeyType thisNode = csTree[nodeIdx];
    KeyType range    = csTree[nodeIdx + 1] - thisNode;
    unsigned level   = treeLevel(range);

    if (level == 0) { return {-1, level}; }

    int siblingIdx = octalDigit(thisNode, level);
    bool siblings  = (csTree[nodeIdx - siblingIdx + 8] == csTree[nodeIdx - siblingIdx] + nodeRange<KeyType>(level - 1));
    if (!siblings) { siblingIdx = -1; }

    return {siblingIdx, level};
}

//! @brief returns 0 for merging, 1 for no-change, 8 for splitting
template<class KeyType>
HOST_DEVICE_FUN int
calculateNodeOp(const KeyType* tree, TreeNodeIndex nodeIdx, const unsigned* counts, unsigned bucketSize)
{
    auto [siblingIdx, level] = siblingAndLevel(tree, nodeIdx);

    if (siblingIdx > 0) // 8 siblings next to each other, node can potentially be merged
    {
        // pointer to first node in sibling group
        auto g = counts + nodeIdx - siblingIdx;
#pragma GCC diagnostic ignored "-Warray-bounds"
        size_t parentCount = size_t(g[0]) + size_t(g[1]) + size_t(g[2]) + size_t(g[3]) + size_t(g[4]) + size_t(g[5]) +
                             size_t(g[6]) + size_t(g[7]);
        bool countMerge = parentCount <= size_t(bucketSize);
        if (countMerge) { return 0; } // merge
    }

    if (counts[nodeIdx] > bucketSize * 512 && level + 3 < maxTreeLevel<KeyType>{}) { return 4096; } // split
    if (counts[nodeIdx] > bucketSize * 64 && level + 2 < maxTreeLevel<KeyType>{}) { return 512; }   // split
    if (counts[nodeIdx] > bucketSize * 8 && level + 1 < maxTreeLevel<KeyType>{}) { return 64; }     // split
    if (counts[nodeIdx] > bucketSize && level < maxTreeLevel<KeyType>{}) { return 8; }              // split

    return 1; // default: do nothing
}

/*! @brief Compute split or fuse decision for each octree node in parallel
 *
 * @tparam    KeyType      32- or 64-bit unsigned integer type
 * @param[in] tree         octree nodes given as SFC codes of length @p nNodes
 *                         needs to satisfy the octree invariants
 * @param[in] counts       output particle counts per node, length = @p nNodes
 * @param[in] nNodes       number of nodes in tree
 * @param[in] bucketSize   maximum particle count per (leaf) node and
 *                         minimum particle count (strictly >) for (implicit) internal nodes
 * @param[out] nodeOps     stores rebalance decision result for each node, length = @p nNodes
 * @return                 true if all nodes are unchanged, false otherwise
 *
 * For each node i in the tree, in nodeOps[i], stores
 *  - 0 if to be merged
 *  - 1 if unchanged,
 *  - 8 if to be split.
 */
template<class KeyType, class LocalIndex>
bool rebalanceDecision(
    const KeyType* tree, const unsigned* counts, TreeNodeIndex nNodes, unsigned bucketSize, LocalIndex* nodeOps)
{
    bool converged = true;

#pragma omp parallel
    {
        bool convergedThread = true;
#pragma omp for
        for (TreeNodeIndex i = 0; i < nNodes; ++i)
        {
            int decision = calculateNodeOp(tree, i, counts, bucketSize);
            if (decision != 1) { convergedThread = false; }

            nodeOps[i] = decision;
        }
        if (!convergedThread) { converged = false; }
    }
    return converged;
}

/*! @brief transform old nodes into new nodes based on opcodes
 *
 * @tparam KeyType    32- or 64-bit integer
 * @param  nodeIndex  the node to process in @p oldTree
 * @param  oldTree    the old tree
 * @param  nodeOps    opcodes per old tree node
 * @param  newTree    the new tree
 */
template<class KeyType>
HOST_DEVICE_FUN void
processNode(TreeNodeIndex nodeIndex, const KeyType* oldTree, const TreeNodeIndex* nodeOps, KeyType* newTree)
{
    KeyType thisNode = oldTree[nodeIndex];
    KeyType range    = oldTree[nodeIndex + 1] - thisNode;
    unsigned level   = treeLevel(range);

    TreeNodeIndex opCode       = nodeOps[nodeIndex + 1] - nodeOps[nodeIndex];
    TreeNodeIndex newNodeIndex = nodeOps[nodeIndex];

    if (opCode == 1) { newTree[newNodeIndex] = thisNode; }
    else if (opCode == 8)
    {
        for (int sibling = 0; sibling < 8; ++sibling)
        {
            newTree[newNodeIndex + sibling] = thisNode + sibling * nodeRange<KeyType>(level + 1);
        }
    }
    else if (opCode > 8)
    {
        unsigned levelDiff = log8ceil(unsigned(opCode));
        for (int sibling = 0; sibling < opCode; ++sibling)
        {
            newTree[newNodeIndex + sibling] = thisNode + sibling * nodeRange<KeyType>(level + levelDiff);
        }
    }
}

/*! @brief split or fuse octree nodes based on node counts relative to bucketSize
 *
 * @param[in]    tree         cornerstone octree
 * @param[out]   newTree      rebalanced cornerstone octree
 * @param[in]    nodeOps      rebalance decision for each node, length @p numNodes(tree) + 1
 *                            will be overwritten
 */
template<class InputVector, class OutputVector>
void rebalanceTree(const InputVector& tree, OutputVector& newTree, TreeNodeIndex* nodeOps)
{
    TreeNodeIndex numNodes = nNodes(tree);

    std::exclusive_scan(nodeOps, nodeOps + numNodes + 1, nodeOps, TreeNodeIndex{0});
    newTree.resize(nodeOps[numNodes] + 1);

#pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < numNodes; ++i)
    {
        processNode(i, tree.data(), nodeOps, newTree.data());
    }
    newTree.back() = tree.back();
}

/*! @brief update the octree with a single rebalance/count step
 *
 * @tparam       KeyType     32- or 64-bit unsigned integer for SFC code
 * @param[in]    keys        local particle SFC keys
 * @param[in]    bucketSize  maximum number of particles per node
 * @param[inout] tree        the octree leaf nodes (cornerstone format)
 * @param[inout] counts      the octree leaf node particle count
 * @param[in]    maxCount    if actual node counts are higher, they will be capped to @p maxCount
 * @return                   true if tree was not modified, false otherwise
 *
 * Remarks:
 *    It is sensible to assume that the bucket size of the tree is much smaller than 2^32,
 *    and thus it is ok to use 32-bit integers for the node counts, because if the node count
 *    happens to be bigger than 2^32 for a node, this node will anyway be divided until the
 *    node count is smaller than the bucket size. We just have to make sure to prevent overflow,
 *    in MPI_Allreduce, therefore, maxCount should be set to 2^32/numRanks - 1 for distributed tree builds.
 */
template<class KeyType>
bool updateOctree(std::span<const KeyType> keys,
                  unsigned bucketSize,
                  std::vector<KeyType>& tree,
                  std::vector<unsigned>& counts,
                  unsigned maxCount = std::numeric_limits<unsigned>::max())
{
    std::vector<TreeNodeIndex> nodeOps(nNodes(tree) + 1);
    bool converged = rebalanceDecision(tree.data(), counts.data(), nNodes(tree), bucketSize, nodeOps.data());

    std::vector<KeyType> newTree;
    rebalanceTree(tree, newTree, nodeOps.data());
    swap(tree, newTree);

    counts.resize(nNodes(tree));
    computeNodeCounts(tree.data(), counts.data(), nNodes(tree), keys, maxCount, true);

    return converged;
}

//! @brief Convenience wrapper for updateOctree. Start from scratch and return a fully converged cornerstone tree.
template<class KeyType>
std::tuple<std::vector<KeyType>, std::vector<unsigned>> computeOctree(
    std::span<const KeyType> keys, unsigned bucketSize, unsigned maxCount = std::numeric_limits<unsigned>::max())
{
    std::vector<KeyType> tree{0, nodeRange<KeyType>(0)};
    std::vector<unsigned> counts{unsigned(keys.size())};

    while (!updateOctree(keys, bucketSize, tree, counts, maxCount))
        ;

    return std::make_tuple(std::move(tree), std::move(counts));
}

/*! @brief update a treelet (sub-octree not spanning full SFC) based on node counts
 *
 * @tparam     KeyType       32- or 64-bit unsigned integer for SFC code
 * @param[in]  treelet       cornerstone key sequence, covering only part of the SFC
 * @param[in]  counts        particle counts per cell in @p treelet, length = treelet.size() - 1
 * @param[in]  bucketSize    rebalance criterion
 * @return                   rebalanced treelet with nodes split where count > @p bucketSize
 *
 * Currently used for rebalancing an SFC range before hand-over to another rank after assignment shift.
 */
template<class KeyType>
std::vector<KeyType>
updateTreelet(std::span<const KeyType> treelet, std::span<const unsigned> counts, unsigned bucketSize)
{
    std::vector<TreeNodeIndex> nodeOps(nNodes(treelet) + 1);
    rebalanceDecision(treelet.data(), counts.data(), nNodes(treelet), bucketSize, nodeOps.data());

    std::vector<KeyType> newTreelet;
    rebalanceTree(treelet, newTreelet, nodeOps.data());

    return newTreelet;
}

/*! @brief create a cornerstone octree around a series of given SFC codes
 *
 * @tparam KeyType        iterator to 32- or 64-bit unsigned integer
 * @param  spanningKeys   input SFC key sequence
 * @return                the cornerstone octree containing all values in the given code sequence
 *                        plus any additional intermediate SFC codes between them required to fulfill
 *                        the cornerstone invariants.
 *
 * Typical application: Generation of a minimal global tree where the input code sequence
 * corresponds to the partitioning of the space filling curve into numMpiRanks intervals.
 *
 *  Requirements on the input sequence [firstCode:lastCode]:
 *      - must start with 0, i.e. *firstCode == 0
 *      - must end with 2^30 or 2^63, i.e. *(lastCode - 1) == nodeRange<CodeType>(0)
 *        with CodeType =- std::decay_t<decltype(*firstCode)>
 *      - must be sorted
 */
template<class KeyType>
std::vector<KeyType> computeSpanningTree(std::span<const KeyType> spanningKeys)
{
    assert(spanningKeys.size() > 1);
    assert(spanningKeys.front() == 0 && spanningKeys.back() == nodeRange<KeyType>(0));

    TreeNodeIndex numIntervals = spanningKeys.size() - 1;

    std::vector<TreeNodeIndex> offsets(numIntervals + 1);
    for (TreeNodeIndex i = 0; i < numIntervals; ++i)
    {
        offsets[i] = spanSfcRange(spanningKeys[i], spanningKeys[i + 1]);
    }

    std::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin(), TreeNodeIndex{0});

    std::vector<KeyType> spanningTree(offsets.back() + 1);
    for (TreeNodeIndex i = 0; i < numIntervals; ++i)
    {
        spanSfcRange(spanningKeys[i], spanningKeys[i + 1], spanningTree.data() + offsets[i]);
    }
    spanningTree.back() = nodeRange<KeyType>(0);

    return spanningTree;
}

} // namespace cstone
