/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Focused octree rebalance on GPUs
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/tree/definitions.h"
#include "cstone/domain/index_ranges.hpp"

namespace cstone
{

template<class KeyType>
extern void rebalanceDecisionEssentialGpu(const KeyType* prefixes,
                                          const TreeNodeIndex* childOffsets,
                                          const TreeNodeIndex* parents,
                                          const unsigned* counts,
                                          const char* macs,
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
                                 const char* macs,
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
extern void rangeCountGpu(gsl::span<const KeyType> leaves,
                          gsl::span<const unsigned> counts,
                          gsl::span<const KeyType> leavesFocus,
                          gsl::span<const TreeNodeIndex> leavesFocusIdx,
                          gsl::span<unsigned> countsFocus);

} // namespace cstone