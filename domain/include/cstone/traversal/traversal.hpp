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

// constexpr int maxCoord = 1u<<maxTreeLevel<KeyType>{};
// KeyType iboxStart = iMorton<KeyType>(collisionBox.xmin(), collisionBox.ymin(), collisionBox.zmin());
// int xmax = collisionBox.xmax();
// int ymax = collisionBox.ymax();
// int zmax = collisionBox.zmax();
// if (xmax == maxCoord) xmax--;
// if (ymax == maxCoord) ymax--;
// if (zmax == maxCoord) zmax--;
// KeyType iboxEnd   = iMorton<KeyType>(xmax, ymax, zmax);

// pair<KeyType> commonBox = smallestCommonBox(iboxStart, iboxEnd);
// int iboxLevel = treeLevel<KeyType>(commonBox[1] - commonBox[0]);

// for (int l = 1; l <= iboxLevel; ++l)
//{
//     int octant = octreeDigit(commonBox[0], l);
//     node = octree.child(node, octant);
// }

// if (octree.isLeaf(node))
//{
//     collisionList.add(node);
//     return;
// }

template<class C, class A>
HOST_DEVICE_FUN void singleTraversal(const TreeNodeIndex* childOffsets, C&& continuationCriterion, A&& endpointAction)
{
    bool descend = continuationCriterion(0);
    if (!descend) return;

    if (childOffsets[0] == 0)
    {
        // root node is already the endpoint
        endpointAction(0);
        return;
    }

    TreeNodeIndex stack[128];
    stack[0] = 0;

    TreeNodeIndex stackPos = 1;
    TreeNodeIndex node     = 0; // start at the root

    do
    {
        for (int octant = 0; octant < 8; ++octant)
        {
            TreeNodeIndex child = childOffsets[node] + octant;
            bool descend        = continuationCriterion(child);
            if (descend)
            {
                if (childOffsets[child] == 0)
                {
                    // endpoint reached with child is a leaf node
                    endpointAction(child);
                }
                else
                {
                    assert(stackPos < 128);
                    stack[stackPos++] = child; // push
                }
            }
        }
        node = stack[--stackPos];

    } while (node != 0); // the root can only be obtained when the tree has been fully traversed
}

/*! @brief Generic dual-traversal of a tree with pairs of indices. Also called simultaneous traversal.
 *
 * Since the continuation criterion and the two endpoint actions for failed/passed criteria are
 * provided as arguments, this function is completely generic and can be used to evaluate MACs
 * for FMM, general collision detection for halo discovery and surface detection.
 *
 *
 * @tparam KeyType         32- or 64-bit unsigned integer
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
template<class TreeType, class MAC, class M2L, class P2P>
void dualTraversal(const TreeType& octree, TreeNodeIndex a, TreeNodeIndex b, MAC&& continuation, M2L&& m2l, P2P&& p2p)
{
    using NodePair = util::array<TreeNodeIndex, 2>;

    if (octree.isLeaf(a) && octree.isLeaf(b))
    {
        if (continuation(a, b)) { p2p(a, b); }
        return;
    }

    NodePair stack[128];
    stack[0] = NodePair{a, b};

    int stackPos = 1;

    auto interact = [&octree, &continuation, &m2l, &p2p, &stackPos](TreeNodeIndex a, TreeNodeIndex b, NodePair* stack_)
    {
        if (continuation(a, b))
        {
            if (octree.isLeaf(a) && octree.isLeaf(b)) { p2p(a, b); }
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

        if ((octree.level(target) < octree.level(source) && !octree.isLeaf(target)) || octree.isLeaf(source))
        {
            int nChildren = (octree.isLeaf(target)) ? 0 : 8;
            for (int octant = 0; octant < nChildren; ++octant)
            {
                interact(octree.child(target, octant), source, stack);
            }
        }
        else
        {
            int nChildren = (octree.isLeaf(source)) ? 0 : 8;
            for (int octant = 0; octant < nChildren; ++octant)
            {
                interact(target, octree.child(source, octant), stack);
            }
        }
    }
}

} // namespace cstone
