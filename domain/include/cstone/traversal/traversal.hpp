/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Generic octree traversal methods
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * Single and dual tree traversal methods are the base algorithms for implementing
 * MAC evaluations, collision and surface detection etc.
 */

#pragma once

#include "cstone/tree/octree.hpp"

namespace cstone
{

template<class C, class A>
HOST_DEVICE_FUN void singleTraversal(const TreeNodeIndex* childOffsets,
                                     const TreeNodeIndex* parents,
                                     C&& continuationCriterion,
                                     A&& endpointAction)
{
    TreeNodeIndex initNode = 0;
    if (!continuationCriterion(initNode)) { return; }

    if (childOffsets[initNode] == 0)
    {
        // initNode (root node) is already the endpoint
        endpointAction(initNode);
        return;
    }

    TreeNodeIndex node = childOffsets[initNode];
    bool backtrack     = false;
    while (node != initNode)
    {
        bool isLeaf  = childOffsets[node] == 0;
        bool descend = not backtrack && continuationCriterion(node);
        // process current node
        if (isLeaf && descend) { endpointAction(node); }

        TreeNodeIndex siblingIdx = (node - 1) % 8;
        // determine next node
        if (!isLeaf && descend) // can we descend?
        {
            node      = childOffsets[node];
            backtrack = false;
        }
        else if (siblingIdx < 7) // can we move to sibling ?
        {
            node++;
            backtrack = false;
        }
        else // move to parent
        {
            node      = parents[(node - 1) / 8];
            backtrack = true;
        }
    }
}

/*! @brief Generic dual-traversal of a tree with pairs of indices. Also called simultaneous traversal.
 *
 * Since the continuation criterion and the two endpoint actions for failed/passed criteria are
 * provided as arguments, this function is completely generic and can be used to evaluate MACs
 * for FMM, general collision detection for halo discovery and surface detection.
 *
 *
 * @tparam MAC             traversal continuation criterion
 * @tparam M2L             endpoint action for nodes that passed @p MAC
 * @tparam P2P             endpoint action for leaf nodes that did not pass @p MAC
 * @param octree           traversable octree
 * @param a                first octree node index for starting the traversal
 * @param b                second start octree node index for starting the traversal
 * @param continuation     Criterion whether or not to continue traversing two nodes
 *                         callable with signature bool(TreeNodeIndex, TreeNodeIndex)
 *                         often, the criterion is some sort of multipole acceptance criterion
 * @param m2l              Multipole-2-local, called each for each node pair during traversal
 *                         that passed @p criterion.
 *                         Callable with signature void(TreeNodeIndex, TreeNodeIndex)
 * @param p2p              Particle-2-particle, called for each pair of leaf nodes during traversal
 *                         that did not pass @p continuation
 */
template<class MAC, class M2L, class P2P>
void dualTraversal(
    const TreeNodeIndex* childOffsets, TreeNodeIndex a, TreeNodeIndex b, MAC&& continuation, M2L&& m2l, P2P&& p2p)
{
    using NodePair = util::array<TreeNodeIndex, 2>;

    auto isLeaf = [childOffsets](TreeNodeIndex idx) { return childOffsets[idx] == 0; };

    if (isLeaf(a) && isLeaf(b))
    {
        if (continuation(a, b)) { p2p(a, b); }
        return;
    }

    NodePair stack[128];
    stack[0] = NodePair{a, b};

    int stackPos = 1;

    auto interact = [isLeaf, &continuation, &m2l, &p2p, &stackPos](TreeNodeIndex a, TreeNodeIndex b, NodePair* stack_)
    {
        if (continuation(a, b))
        {
            if (isLeaf(a) && isLeaf(b)) { p2p(a, b); }
            else
            {
                assert(stackPos < 128);
                stack_[stackPos++] = NodePair{a, b};
            }
        }
        else { m2l(a, b); }
    };

    while (stackPos > 0)
    {
        NodePair nodePair    = stack[--stackPos];
        TreeNodeIndex target = nodePair[0];
        TreeNodeIndex source = nodePair[1];

        // target < source means level(target) <= level(source)
        if ((target < source && !isLeaf(target)) || isLeaf(source)) // subdivide target
        {
            int nChildren = isLeaf(target) ? 0 : 8;
            for (int octant = 0; octant < nChildren; ++octant)
            {
                interact(childOffsets[target] + octant, source, stack);
            }
        }
        else // subdivide source
        {
            int nChildren = isLeaf(source) ? 0 : 8;
            for (int octant = 0; octant < nChildren; ++octant)
            {
                interact(target, childOffsets[source] + octant, stack);
            }
        }
    }
}

} // namespace cstone
