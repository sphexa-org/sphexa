/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Focused octree rebalance on GPUs
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <span>

#include "cstone/tree/definitions.h"
#include "cstone/domain/index_ranges.hpp"

namespace cstone
{

template<class KeyType>
extern void rebalanceDecisionEssentialGpu(const KeyType* prefixes,
                                          const TreeNodeIndex* childOffsets,
                                          const TreeNodeIndex* parents,
                                          const unsigned* counts,
                                          const uint8_t* macs,
                                          KeyType focusStart,
                                          KeyType focusEnd,
                                          unsigned bucketSize,
                                          TreeNodeIndex* nodeOps,
                                          TreeNodeIndex numNodes);

/*! @brief Take decision how to refine nodes based on Macs
 *
 * @param[in]  prefixes       WS-SFC key of each node, length numNodes
 * @param[in]  macs           mac evaluation result flags, length numNodes
 * @param[in]  l2i            translates indices in [0:numLeafNodes] to [0:numNodes] to access prefixes and macs
 * @param[in]  numLeafNodes   number of leaf nodes
 * @param[in]  focus          index range within [0:numLeafNodes] that corresponds to nodes in focus
 * @param[out] nodeOps        output refinement decision per leaf node
 */
template<class KeyType>
extern void macRefineDecisionGpu(const KeyType* prefixes,
                                 const uint8_t* macs,
                                 const TreeNodeIndex* l2i,
                                 TreeNodeIndex numLeafNodes,
                                 TreeIndexPair focus,
                                 TreeNodeIndex* nodeOps);

template<class KeyType>
extern bool protectAncestorsGpu(const KeyType*, const TreeNodeIndex*, TreeNodeIndex*, TreeNodeIndex);

template<class KeyType>
extern ResolutionStatus enforceKeysGpu(const KeyType* forcedKeys,
                                       TreeNodeIndex numForcedKeys,
                                       const KeyType* nodeKeys,
                                       const TreeNodeIndex* childOffsets,
                                       const TreeNodeIndex* parents,
                                       TreeNodeIndex* nodeOps);

//! @brief see CPU version
template<class KeyType>
extern void rangeCountGpu(std::span<const KeyType> leaves,
                          std::span<const unsigned> counts,
                          std::span<const KeyType> leavesFocus,
                          std::span<const TreeNodeIndex> leavesFocusIdx,
                          std::span<unsigned> countsFocus);

} // namespace cstone