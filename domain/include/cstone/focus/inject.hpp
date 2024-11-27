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
 * @brief SFC key injection into cornerstone arrays to enforce the presence of specified keys
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/tree/csarray.hpp"
#include "cstone/tree/csarray_gpu.h"
#include "cstone/tree/octree.hpp"
#include "cstone/primitives/primitives_gpu.h"

namespace cstone
{

/*! @brief inject specified keys into a cornerstone leaf tree
 *
 * @tparam KeyType    32- or 64-bit unsigned integer
 * @param[inout] tree   cornerstone octree
 * @param[in]    keys   list of SFC keys to insert
 *
 * This function needs to insert more than just @p keys, due the cornerstone
 * invariant of consecutive nodes always having a power-of-8 difference.
 * This means that each subdividing a node, all 8 children always have to be added.
 */
template<class KeyType, class Alloc>
void injectKeys(std::vector<KeyType, Alloc>& tree, gsl::span<const KeyType> keys)
{
    std::vector<KeyType> spanningKeys(keys.begin(), keys.end());
    spanningKeys.push_back(0);
    spanningKeys.push_back(nodeRange<KeyType>(0));
    std::sort(begin(spanningKeys), end(spanningKeys));
    auto uit = std::unique(begin(spanningKeys), end(spanningKeys));
    spanningKeys.erase(uit, end(spanningKeys));

    // spanningTree is a list of all the missing nodes needed to resolve the mandatory keys
    auto spanningTree = computeSpanningTree<KeyType>(spanningKeys);
    tree.reserve(tree.size() + spanningTree.size());

    // spanningTree is now inserted into newLeaves
    std::copy(begin(spanningTree), end(spanningTree), std::back_inserter(tree));

    // cleanup, restore invariants: sorted-ness, no-duplicates
    std::sort(begin(tree), end(tree));
    uit = std::unique(begin(tree), end(tree));
    tree.erase(uit, end(tree));
}

/*! @brief inject specified keys into a gpu cornerstone leaf tree
 *
 * @tparam
 * @param[inout] leaves       cornerstone leaf key array
 * @param[in]    keys         SFC keys to inject into @p leaves
 * @param[-]     keyScratch   temporary space, content will be overwritten and resized
 * @param[-]     spanOps      temporary space, content will be overwritten and resized
 * @param[-]     spanOpsScan  temporary space, content will be overwritten and resized
 *
 * Injects keys into leaves, sorts and adds missing keys inbetween to satisfy power-of-8 differences
 * between consecutive keys.
 */
template<class KeyType>
void injectKeysGpu(DeviceVector<KeyType>& leaves,
                   gsl::span<const KeyType> keys,
                   DeviceVector<KeyType>& keyScratch,
                   DeviceVector<TreeNodeIndex>& spanOps,
                   DeviceVector<TreeNodeIndex>& spanOpsScan)
{
    reallocate(leaves, leaves.size() + keys.size(), 1.0);
    memcpyD2D(keys.data(), keys.size(), leaves.data() + leaves.size() - keys.size());

    reallocateDestructive(keyScratch, leaves.size(), 1.0);
    sortGpu(rawPtr(leaves), rawPtr(leaves) + leaves.size(), rawPtr(keyScratch));

    reallocateDestructive(spanOps, leaves.size(), 1.0);
    reallocateDestructive(spanOpsScan, leaves.size(), 1.0);

    countSfcGapsGpu(leaves.data(), nNodes(leaves), spanOps.data());
    exclusiveScanGpu(spanOps.data(), spanOps.data() + leaves.size(), spanOpsScan.data());

    TreeNodeIndex numNodesGap;
    memcpyD2H(spanOpsScan.data() + leaves.size() - 1, 1, &numNodesGap);

    reallocateDestructive(keyScratch, numNodesGap + 1, 1.0);
    fillSfcGapsGpu(leaves.data(), nNodes(leaves), spanOpsScan.data(), keyScratch.data());
    swap(leaves, keyScratch);
}

} // namespace cstone
