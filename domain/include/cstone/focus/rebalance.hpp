/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Focused octree rebalance
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/traversal/boxoverlap.hpp"
#include "cstone/tree/csarray.hpp"
#include "cstone/tree/octree.hpp"

namespace cstone
{
/*! @brief combines the particle count and multipole criteria for rebalancing
 *
 * @return
 *      - 0 if node to be merged
 *      - 1 if node to stay unchanged
 *      - 8 if node to be split
 */
template<class KeyType>
HOST_DEVICE_FUN int mergeCountAndMacOp(TreeNodeIndex nodeIdx,
                                       const KeyType* nodeKeys,
                                       const TreeNodeIndex* childOffsets,
                                       const TreeNodeIndex* parents,
                                       const unsigned* counts,
                                       const uint8_t* macs,
                                       KeyType focusStart,
                                       KeyType focusEnd,
                                       unsigned bucketSize)
{
    TreeNodeIndex siblingGroup = (nodeIdx - 1) / 8;
    TreeNodeIndex parent       = nodeIdx ? parents[siblingGroup] : 0;
    KeyType nodeKey            = nodeKeys[nodeIdx];
    unsigned level             = decodePrefixLength(nodeKeys[nodeIdx]) / 3;

    if (nodeIdx)
    {
        bool countMerge = counts[parent] <= bucketSize;
        bool macMerge   = macs[parent] == 0;

        KeyType firstGroupKey = decodePlaceholderBit(nodeKeys[parent]);
        KeyType lastGroupKey  = firstGroupKey + 8 * nodeRange<KeyType>(level);
        // inFringe: nodeIdx not in focus, but at least one sibling is in the focus
        // in that case we cannot remove the nodes based on a MAC criterion
        bool inFringe = overlapTwoRanges(firstGroupKey, lastGroupKey, focusStart, focusEnd);

        if (countMerge || (macMerge && !inFringe)) { return 0; } // merge
    }

    KeyType nodeStart = decodePlaceholderBit(nodeKey);
    bool isLeaf       = childOffsets[nodeIdx] == 0;
    bool inFocus      = (nodeStart >= focusStart && nodeStart < focusEnd);
    if (isLeaf && level < maxTreeLevel<KeyType>{} && counts[nodeIdx] > bucketSize && (macs[nodeIdx] || inFocus))
    {
        return 8; // split
    }

    return 1; // default: do nothing
}

//! @brief refine nodes based on Mac only
template<class KeyType>
HOST_DEVICE_FUN int macRefineOp(KeyType nodeKey, char mac)
{
    unsigned level = decodePrefixLength(nodeKey) / 3;
    if (level < maxTreeLevel<KeyType>{} && mac) { return 8; }
    return 1;
}

/*! @brief Overrides a 0-value of nodeOps[nodeIdx] if @p nodeIdx is the left-most descendant of a non-zero ancestor
 *
 * @tparam KeyType
 * @param nodeIdx   index of a leaf node to check
 * @param prefixes  node SFC keys in placeholder-bit format
 * @param parents   parent node indices per group of 8 siblings
 * @param nodeOps   node transformation operation in {0, 1, 8}
 * @return          0 or nodeOps[closestNonZeroAncestor]
 *
 * If @p nodeIdx is the left-most descendant of the closest ancestor with a non-zero nodeOps value, return
 * the nodeOps value of the ancestor, 0 otherwise
 */
template<class KeyType>
HOST_DEVICE_FUN TreeNodeIndex
nzAncestorOp(TreeNodeIndex nodeIdx, const KeyType* prefixes, const TreeNodeIndex* parents, const TreeNodeIndex* nodeOps)
{
    if (nodeIdx == 0) { return nodeOps[0]; }

    TreeNodeIndex closestNonZeroAncestor = nodeIdx;
    while (nodeOps[closestNonZeroAncestor] == 0)
    {
        closestNonZeroAncestor = parents[(closestNonZeroAncestor - 1) / 8];
    }

    if (decodePlaceholderBit(prefixes[nodeIdx]) == decodePlaceholderBit(prefixes[closestNonZeroAncestor]))
    {
        return nodeOps[closestNonZeroAncestor];
    }

    return 0;
}

/*! @brief Compute locally essential split or fuse decision for each octree node in parallel
 *
 * @tparam    KeyType          32- or 64-bit unsigned integer type
 * @param[in] nodeKeys         warren-salmon SFC keys for each tree node, length = @p numNodes
 * @param[in] childOffsets     node index of first child (0 identifies a leaf), length = @p numNodes
 * @param[in] parents          parent node index for each group of 8 siblings, length = (numNodes-1) / 8
 * @param[in] counts           particle count of each tree node, length = @p numNodes
 * @param[in] macs             multipole pass or fail per node, length = @p numNodes
 * @param[in] focusStart       first focus SFC key
 * @param[in] focusEnd         last focus SFC key
 * @param[in] bucketSize       maximum particle count per (leaf) node and
 *                             minimum particle count (strictly >) for (implicit) internal nodes
 * @param[out] nodeOps         stores rebalance decision result for each node, length = @p numNodes
 *                             only leaf nodes will be set, internal nodes are ignored
 * @return                     true if converged, false otherwise
 *
 * For each leaf node i in the tree, in nodeOps[i], stores
 *  - 0 if to be merged
 *  - 1 if unchanged,
 *  - 8 if to be split.
 */
template<class KeyType>
void rebalanceDecisionEssential(std::span<const KeyType> nodeKeys,
                                const TreeNodeIndex* childOffsets,
                                const TreeNodeIndex* parents,
                                const unsigned* counts,
                                const uint8_t* macs,
                                KeyType focusStart,
                                KeyType focusEnd,
                                unsigned bucketSize,
                                TreeNodeIndex* nodeOps)
{
#pragma omp parallel for
    for (std::size_t i = 0; i < nodeKeys.size(); ++i)
    {
        nodeOps[i] = mergeCountAndMacOp(i, nodeKeys.data(), childOffsets, parents, counts, macs, focusStart, focusEnd,
                                        bucketSize);
    }
}

template<class KeyType>
bool protectAncestors(std::span<const KeyType> nodeKeys, const TreeNodeIndex* parents, TreeNodeIndex* nodeOps)
{
    int numChanges = 0;
#pragma omp parallel for reduction(+ : numChanges)
    for (std::size_t i = 0; i < nodeKeys.size(); ++i)
    {
        int opDecision = nzAncestorOp(i, nodeKeys.data(), parents, nodeOps);

        if (opDecision != 1) { numChanges++; }
        nodeOps[i] = opDecision;
    }
    return numChanges == 0;
}

enum class ResolutionStatus : int
{
    //! @brief required SFC keys present in tree, no action needed
    converged,
    //! @brief required SFC keys already present in tree, but had to cancel rebalance-merge operations
    cancelMerge,
    //! @brief subsequent rebalance can resolve the required SFC key by subdividing the closest node
    rebalance,
    //! @brief subsequent rebalance cannot resolve the required SFC key with subdividing the closest node
    failed
};

template<class KeyType>
HOST_DEVICE_FUN ResolutionStatus enforceKeySingle(KeyType key,
                                                  const KeyType* nodeKeys,
                                                  const TreeNodeIndex* childOffsets,
                                                  const TreeNodeIndex* parents,
                                                  TreeNodeIndex* nodeOps)
{
    if (key == 0 || key == nodeRange<KeyType>(0)) { return ResolutionStatus::converged; }

    ResolutionStatus status = ResolutionStatus::converged;

    KeyType nodeKeyWant    = makePrefix(key);
    TreeNodeIndex nodeIdx  = containingNode(nodeKeyWant, nodeKeys, childOffsets);
    KeyType nodeKeyHave    = nodeKeys[nodeIdx];
    unsigned nodeLevelHave = decodePrefixLength(nodeKeyHave) / 3;

    // need to undo merges of all supporting ancestors if the closest tree node would be merged
    // or the mandatory key is not there
    bool trySplit   = nodeKeyHave != nodeKeyWant && nodeLevelHave < maxTreeLevel<KeyType>{};
    bool undoMerges = nodeOps[nodeIdx] == 0 || trySplit;
    if (undoMerges && nodeIdx > 0)
    {
        status = ResolutionStatus::cancelMerge;
        // make sure no supporting ancestor of nodeIdx will be merged
        TreeNodeIndex parent = nodeIdx;
        do
        {
            parent                     = parents[(parent - 1) / 8];
            TreeNodeIndex firstSibling = childOffsets[parent];
            for (TreeNodeIndex i = firstSibling; i < firstSibling + eightSiblings; ++i)
            {
                if (nodeOps[i] == 0) { nodeOps[i] = 1; }
            }
        } while (parent != 0);
    }

    if (trySplit)
    {
        int keyPos = lastNzPlace(key);

        // only add 1 level, otherwise nodes can be added in a non-peer area,
        // exceeding the resolution of the global tree, which will result in a failure to compute
        // exact particle counts for those nodes
        constexpr int maxAddLevels = 1;
        int levelDiff              = keyPos - nodeLevelHave;
        if (levelDiff > maxAddLevels) { status = ResolutionStatus::failed; }
        else { status = ResolutionStatus::rebalance; }

        levelDiff        = stl::min(levelDiff, maxAddLevels);
        nodeOps[nodeIdx] = stl::max(nodeOps[nodeIdx], 1 << (3 * levelDiff));
    }

    return status;
}

template<class KeyType>
ResolutionStatus enforceKeys(std::span<const KeyType> mandatoryKeys,
                             const KeyType* nodeKeys,
                             const TreeNodeIndex* childOffsets,
                             const TreeNodeIndex* parents,
                             TreeNodeIndex* nodeOps)
{
    ResolutionStatus status = ResolutionStatus::converged;

    for (KeyType key : mandatoryKeys)
    {
        status = std::max(enforceKeySingle(key, nodeKeys, childOffsets, parents, nodeOps), status);
    }
    return status;
}

/*! @brief count particles inside specified ranges of a cornerstone leaf tree
 *
 * @tparam KeyType             32- or 64-bit unsigned integer
 * @param[in]  leaves          global octree leaves, cornerstone SFC key sequence
 * @param[in]  counts          global particle counts of @p leaves, length = length(leaves) - 1
 * @param[in]  leavesFocus     focused octree leaves (LET), cornerstone SFC key sequence
 * @param[in]  leavesFocusIdx  list of cell indices of @p leavesFocus to extract counts for
 * @param[out] countsFocus     output counts for @p leavesFocus, length = length(leavesFocus) - 1
 */
template<class KeyType>
void rangeCount(std::span<const KeyType> leaves,
                std::span<const unsigned> counts,
                std::span<const KeyType> leavesFocus,
                std::span<const TreeNodeIndex> leavesFocusIdx,
                std::span<unsigned> countsFocus)
{
#pragma omp parallel for
    for (size_t i = 0; i < leavesFocusIdx.size(); ++i)
    {
        TreeNodeIndex leafIdx = leavesFocusIdx[i];
        KeyType startKey      = leavesFocus[leafIdx];
        KeyType endKey        = leavesFocus[leafIdx + 1];

        size_t startIdx = findNodeBelow(leaves.data(), leaves.size(), startKey);
        size_t endIdx   = findNodeAbove(leaves.data(), leaves.size(), endKey);
        assert(startKey == leaves[startIdx] && endKey == leaves[endIdx]);

        uint64_t globCount   = std::accumulate(counts.begin() + startIdx, counts.begin() + endIdx, uint64_t(0));
        countsFocus[leafIdx] = std::min(uint64_t(std::numeric_limits<unsigned>::max()), globCount);
    }
}

} // namespace cstone
