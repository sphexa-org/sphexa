/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Functions for upsweeping child data to internal nodes
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#include <array>

#include "cstone/primitives/math.hpp"
#include "cstone/tree/octree.hpp"
#include "cstone/util/tuple_util.hpp"

namespace cstone::ijloop
{

namespace detail
{

template<class TransformOp, class BinaryOp, class Init, class Input, class Output>
__global__ void upsweepAccumulateLeafNodes(const TreeNodeIndex* __restrict__ leafToInternal,
                                           const TreeNodeIndex numLeafNodes,
                                           const LocalIndex* __restrict__ layout,
                                           const Init init,
                                           TransformOp transformOp,
                                           BinaryOp binaryOp,
                                           const Input input,
                                           const Output output)
{
    const TreeNodeIndex leafIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leafIdx >= numLeafNodes) return;

    const TreeNodeIndex nodeIdx = leafToInternal[leafIdx];
    auto accum                  = init;
    for (LocalIndex i = layout[leafIdx]; i < layout[leafIdx + 1]; ++i)
        accum = binaryOp(accum, transformOp(util::tupleMap([&](const auto* ptr) { return ptr[i]; }, input)));
    util::for_each_tuple([&](auto* ptr, auto value) { ptr[nodeIdx] = value; }, output, accum);
}

template<class BinaryOp, class Init, class Output>
__global__ void upsweepAccumulateInternalNodes(const TreeNodeIndex firstNode,
                                               const TreeNodeIndex lastNode,
                                               const TreeNodeIndex* __restrict__ childOffsets,
                                               const Init init,
                                               BinaryOp binaryOp,
                                               const Output output)
{
    const TreeNodeIndex nodeIdx = blockIdx.x * blockDim.x + threadIdx.x + firstNode;
    if (nodeIdx >= lastNode) return;

    const TreeNodeIndex firstChild = childOffsets[nodeIdx];
    if (!firstChild) return;

    auto accum = init;
    for (TreeNodeIndex childIdx = firstChild; childIdx < firstChild + eightSiblings; ++childIdx)
        accum = binaryOp(accum, util::tupleMap([&](const auto* ptr) { return ptr[childIdx]; }, output));
    util::for_each_tuple([&](auto* ptr, auto value) { ptr[nodeIdx] = value; }, output, accum);
}

} // namespace detail

/*! upsweep operation for an octree
 *
 * This function performs an upsweep (bottom-up reduction) over an octree, accumulating data from child nodes
 * to their respective parent internal nodes. It first processes all leaf nodes using the provided transform
 * and binary operation, then iteratively processes internal nodes level by level up to the root.
 *
 * @param[in]  tree        octree view containing structure and metadata
 * @param[in]  init        tuple of initial values for the accumulation
 * @param[in]  transformOp unary operation to transform input data at the leaves
 * @param[in]  binaryOp    binary operation to combine values during accumulation
 * @param[in]  input       tuple of input pointers for leaf data
 * @param[out] output      tuple of output pointers for accumulated results
 */
template<class Tc, class KeyType, class TransformOp, class BinaryOp, class... In, class... Out>
void upsweep(const OctreeNsView<Tc, KeyType>& tree,
             const std::tuple<Out...>& init,
             TransformOp&& transformOp,
             BinaryOp&& binaryOp,
             const std::tuple<In*...> input,
             const std::tuple<Out*...> output)
{
    constexpr unsigned numThreads = 256;

    if (tree.numLeafNodes)
    {
        detail::upsweepAccumulateLeafNodes<<<iceil(tree.numLeafNodes, numThreads), numThreads>>>(
            tree.leafToInternal, tree.numLeafNodes, tree.layout, init, std::forward<TransformOp>(transformOp),
            std::forward<BinaryOp>(binaryOp), input, output);
        checkGpuErrors(cudaGetLastError());
    }

    std::array<TreeNodeIndex, maxTreeLevel<KeyType>() + 2> levelRange;
    memcpyD2H(tree.levelRange, levelRange.size(), levelRange.data());

    for (int level = maxTreeLevel<KeyType>() - 1; level >= 0; --level)
    {
        const TreeNodeIndex firstNode = levelRange[level];
        const TreeNodeIndex lastNode  = levelRange[level + 1];
        const TreeNodeIndex numNodes  = lastNode - firstNode;
        if (numNodes)
        {
            detail::upsweepAccumulateInternalNodes<<<iceil(numNodes, numThreads), numThreads>>>(
                firstNode, lastNode, tree.childOffsets, init, std::forward<BinaryOp>(binaryOp), output);
            checkGpuErrors(cudaGetLastError());
        }
    }
}

} // namespace cstone::ijloop
