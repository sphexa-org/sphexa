/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Functions for finding peer ranks for point to point communication in global domains
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/traversal/macs.hpp"
#include "cstone/domain/domaindecomp.hpp"

namespace cstone
{

/*! @brief find peer ranks based on a multipole acceptance criterion and dual tree traversal
 *
 * @tparam T            float or double
 * @tparam KeyType      32- or 64-bit unsigned integer
 * @param myRank        find peers for the globally assigned SFC segment with index myRank
 * @param assignment    Decomposition of the global SFC into segments
 * @param domainTree    octree built on top of the global cornerstone leaves
 * @param box           global coordinate bounding box
 * @param invThetaEff   1/theta + s, effective inverse opening parameter
 * @return              list of segment indices (i.e. "ranks") that contain tree leaf nodes
 *                      that fail the MAC paired with at least one tree leaf node inside
 *                      the @p myRank segment. This list contains at least the segments
 *                      at the surface of the @p myRank segment and possibly additional
 *                      segments for low opening angles and/or low global resolution in
 *                      @p domainTree.
 *
 * Note: This function guarantees mutuality, if rank A identifies B as peer, then also
 *       rank B will have A as peer
 *
 * Except for @p myRank, this function acts on data that is identical on all MPI ranks and
 * doesn't need to do any communication.
 */
template<class T, class KeyType>
std::vector<int> findPeersMac(int myRank,
                              const SfcAssignment<KeyType>& assignment,
                              OctreeView<const KeyType> domainTree,
                              const Box<T>& box,
                              float invThetaEff,
                              const bool disableMixD = false)
{
    KeyType domainStart = assignment[myRank];
    KeyType domainEnd   = assignment[myRank + 1];
    std::cout << "[findPeersMac] myRank: " << myRank << " domainStart: " << std::oct << domainStart << " domainEnd: " << domainEnd
              << std::dec << std::endl;
    const auto mixDBits = getBoxMixDimensionBits<T, KeyType, Box<T>>(box);
    const auto maxMixDBits = std::max({mixDBits.bx, mixDBits.by, mixDBits.bz});
    const auto minMixDBits = std::min({mixDBits.bx, mixDBits.by, mixDBits.bz});
    const bool useMixD = !disableMixD && (mixDBits.bx != maxTreeLevel<KeyType>{} ||
                          mixDBits.by != maxTreeLevel<KeyType>{} ||
                          mixDBits.bz != maxTreeLevel<KeyType>{});

    int maxCoord   = 1u << maxTreeLevel<KeyType>{};
    float roundOff = 1 + 1e-6; // ensure that peers are picked up in case of a numerical tie
    auto ellipse   = Vec3<T>{box.ilx(), box.ily(), box.ilz()} * box.maxExtent() * invThetaEff * roundOff;
    // auto ellipse = Vec3<T>{
    //     static_cast<T>(1 << mixDBits.bx),
    //     static_cast<T>(1 << mixDBits.by),
    //     static_cast<T>(1 << mixDBits.bz)
    // } * static_cast<T>(1 << maxTreeLevel<KeyType>{}) * static_cast<T>(invThetaEff) * static_cast<T>(roundOff);
    // TODO(iomaganaris): ellipse calculation probably needs fixing according to the the level of the tree because the boxes have different lengths
    // auto ellipse   = Vec3<T>{
    //     box.ilx() * T(1 << (maxTreeLevel<KeyType>{} - mixDBits.bx)),
    //     box.ily() * T(1 << (maxTreeLevel<KeyType>{} - mixDBits.by)),
    //     box.ilz() * T(1 << (maxTreeLevel<KeyType>{} - mixDBits.bz))
    // } * box.maxExtent() * invThetaEff * roundOff;
    auto pbc_t     = BoundaryType::periodic;
    // std::cout << "[findPeersMac] is periodic in x: " << (box.boundaryX() == pbc_t) << " y: " << (box.boundaryY() == pbc_t)
    //           << " z: " << (box.boundaryZ() == pbc_t) << std::endl;
    auto pbc       = useMixD ? Vec3<int>{static_cast<int>((box.boundaryX() == pbc_t) * (1u << mixDBits.bx)),
                                         static_cast<int>((box.boundaryY() == pbc_t) * (1u << mixDBits.by)),
                                         static_cast<int>((box.boundaryZ() == pbc_t) * (1u << mixDBits.bz))}
                             : Vec3<int>{static_cast<int>(box.boundaryX() == pbc_t),
                                         static_cast<int>(box.boundaryY() == pbc_t),
                                         static_cast<int>(box.boundaryZ() == pbc_t)} * maxCoord;

    auto crossFocusPairs = [domainStart, domainEnd, ellipse, pbc, &tree = domainTree, &mixDBits, useMixD, &box, &invThetaEff, myRank](TreeNodeIndex a, TreeNodeIndex b)
    {
        auto [ka1, ka2]    = decodePlaceholderBit2K(tree.prefixes[a]);
        auto [kb1, kb2]    = decodePlaceholderBit2K(tree.prefixes[b]);
        bool aFocusOverlap = overlapTwoRanges(domainStart, domainEnd, ka1, ka2);
        bool bInFocus      = containedIn(kb1, kb2, domainStart, domainEnd);
        // node a has to overlap/be contained in the focus, while b must not be inside it
        if (!aFocusOverlap || bInFocus) { return false; }

        IBox aBox = useMixD ? sfcIBox(sfcMixDKey(ka1), maxTreeLevel<KeyType>{}-treeLevel(ka2 - ka1), mixDBits.bx, mixDBits.by, mixDBits.bz) : sfcIBox(sfcKey(ka1), treeLevel(ka2 - ka1));
        if (aBox.xmax() == aBox.xmin() &&
            aBox.ymax() == aBox.ymin() &&
            aBox.zmax() == aBox.zmin())
        {
            return false; // skip empty boxes
        }
        IBox bBox = useMixD ? sfcIBox(sfcMixDKey(kb1), maxTreeLevel<KeyType>{}-treeLevel(kb2 - kb1), mixDBits.bx, mixDBits.by, mixDBits.bz) : sfcIBox(sfcKey(kb1), treeLevel(kb2 - kb1));
        if (bBox.xmax() == bBox.xmin() &&
            bBox.ymax() == bBox.ymin() &&
            bBox.zmax() == bBox.zmin())
        {
            return false; // skip empty boxes
        }
        auto [aSourceCenter, aSourceSize] = centerAndSize<KeyType>(aBox, box);
        T a_size = std::sqrt(4 * aSourceSize[0] * aSourceSize[0] +
                     4 * aSourceSize[1] * aSourceSize[1] +
                     4 * aSourceSize[2] * aSourceSize[2]);
        T lA   = T(2) * a_size;
        // T lA   = T(2) * max(aSourceSize);
        // T lA   = a_size;
        T macA = lA * invThetaEff;
        T mac2a = macA * macA;
        auto [bTargetCenter, bTargetSize] = centerAndSize<KeyType>(bBox, box);
        T b_size = std::sqrt(4 * bTargetSize[0] * bTargetSize[0] +
                     4 * bTargetSize[1] * bTargetSize[1] +
                     4 * bTargetSize[2] * bTargetSize[2]);
        T lB   = T(2) * b_size;
        // T lB   = T(2) * max(bTargetSize);
        // T lB   = b_size;
        T macB = lB * invThetaEff;
        T mac2b = macB * macB;
        bool violatesMacA = evaluateMacPbc(aSourceCenter, mac2a, bTargetCenter, bTargetSize, box);
        bool violatesMacB = evaluateMacPbc(bTargetCenter, mac2b, aSourceCenter, aSourceSize, box);
        // if (violatesMacA != violatesMacB)
        // {
        //     // std::cout << "[findPeersMac] myRank: " << myRank << " Box A: [" << aBox.xmin() << "," << aBox.xmax() << "] x [" << aBox.ymin() << "," << aBox.ymax() << "] x ["
        //     //           << aBox.zmin() << "," << aBox.zmax() << "] Box B: [" << bBox.xmin() << "," << bBox.xmax() << "] x [" << bBox.ymin() << "," << bBox.ymax() << "] x ["
        //     //           << bBox.zmin() << "," << bBox.zmax() << "] Center A: " << aSourceCenter[0] << " " << aSourceCenter[1] << " " << aSourceCenter[2] << " size: " << aSourceSize[0] << " " << aSourceSize[1]
        //     //           << " " << aSourceSize[2] << " mac: " << macA << " Center B: " << bTargetCenter[0] << " " << bTargetCenter[1] << " " << bTargetCenter[2] << " size: " << bTargetSize[0] << " " << bTargetSize[1]
        //     //           << " " << bTargetSize[2] << " mac: " << macB;
        //     // throw std::runtime_error("Mac is not symmetric");
        //     // std::cout << " Mac is not symmetric" << std::endl;
        // }
        // if ((violatesMacA && minMacMutualInt(aBox, bBox, ellipse, pbc)) || (!violatesMacA && !minMacMutualInt(aBox, bBox, ellipse, pbc))) {
        //     // std::cout << "[findPeersMac] myRank: " << myRank << " Box A: [" << aBox.xmin() << "," << aBox.xmax() << "] x [" << aBox.ymin() << "," << aBox.ymax() << "] x ["
        //     //           << aBox.zmin() << "," << aBox.zmax() << "]" << std::endl;
        //     // std::cout << "[findPeersMac] myRank: " << myRank << " Box B: [" << bBox.xmin() << "," << bBox.xmax() << "] x [" << bBox.ymin() << "," << bBox.ymax() << "] x ["
        //     //           << bBox.zmin() << "," << bBox.zmax() << "]" << std::endl;
        //     // std::cout << "[findPeersMac] myRank: " << myRank << " Center A: " << aSourceCenter[0] << " " << aSourceCenter[1] << " " << aSourceCenter[2] << " size: " << aSourceSize[0] << " " << aSourceSize[1]
        //     //           << " " << aSourceSize[2] << " mac: " << macA << std::endl;
        //     // std::cout << "[findPeersMac] myRank: " << myRank << " Center B: " << bTargetCenter[0] << " " << bTargetCenter[1] << " " << bTargetCenter[2] << " size: " << bTargetSize[0] << " " << bTargetSize[1]
        //     //           << " " << bTargetSize[2] << " mac: " << macB << std::endl;
        //     throw std::runtime_error("Logic error in findPeersMac");
        // }

        return !minMacMutualInt(aBox, bBox, ellipse, pbc); // Return value must be false for m2l and true for p2p
    };

    auto m2l = [](TreeNodeIndex, TreeNodeIndex) {};

    std::vector<int> peerRanks(assignment.numRanks(), 0);
    auto p2p = [&domainTree, &assignment, &peerRanks](TreeNodeIndex a, TreeNodeIndex b)
    {
        int peerRank = assignment.findRank(decodePlaceholderBit(domainTree.prefixes[b]));
        if (peerRanks[peerRank] == 0) { peerRanks[peerRank] = 1; }
    };

    std::vector<KeyType> spanningNodeKeys(spanSfcRange(domainStart, domainEnd) + 1);
    spanSfcRange(domainStart, domainEnd, spanningNodeKeys.data());
    spanningNodeKeys.back() = domainEnd;

#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < spanningNodeKeys.size() - 1; ++i)
    {
        TreeNodeIndex nodeIdx =
            locateNode(spanningNodeKeys[i], spanningNodeKeys[i + 1], domainTree.prefixes, domainTree.levelRange);
        dualTraversal(domainTree.childOffsets, domainTree.empty, nodeIdx, 0, crossFocusPairs, m2l, p2p);
    }

    // for (int i = 0; i < int(peerRanks.size()); ++i)
    // {
    //     if (peerRanks[i]) { std::cout << i << " "; }
    // }
    // std::cout << std::endl;

    std::vector<int> ret;
    for (int i = 0; i < int(peerRanks.size()); ++i)
    {
        if (peerRanks[i]) { ret.push_back(i); }
    }

    return ret;
}

//! @brief Args identical to findPeersMac, but implemented with single tree traversal for comparison
template<class KeyType, class T>
std::vector<int> findPeersMacStt(int myRank,
                                 const SfcAssignment<KeyType>& assignment,
                                 const Octree<KeyType>& octree,
                                 const Box<T>& box,
                                 float invThetaEff,
                                 const bool disableMixD = false)
{
    KeyType domainStart     = assignment[myRank];
    KeyType domainEnd       = assignment[myRank + 1];
    const KeyType* leaves   = octree.treeLeaves().data();
    TreeNodeIndex firstLeaf = findNodeAbove(leaves, octree.numLeafNodes(), domainStart);
    TreeNodeIndex lastLeaf  = findNodeAbove(leaves, octree.numLeafNodes(), domainEnd);

    int maxCoord = 1u << maxTreeLevel<KeyType>{};
    auto ellipse = Vec3<T>{box.ilx(), box.ily(), box.ilz()} * box.maxExtent() * invThetaEff;
    auto pbc_t   = BoundaryType::periodic;
    auto pbc     = Vec3<int>{box.boundaryX() == pbc_t, box.boundaryY() == pbc_t, box.boundaryZ() == pbc_t} * maxCoord;

    std::vector<int> peers(assignment.numRanks());

    const auto mixDBits = getBoxMixDimensionBits<T, KeyType, Box<T>>(box);
    const bool useMixD = !disableMixD && (mixDBits.bx != maxTreeLevel<KeyType>{} ||
                          mixDBits.by != maxTreeLevel<KeyType>{} ||
                          mixDBits.bz != maxTreeLevel<KeyType>{});

#pragma omp parallel for
    for (TreeNodeIndex i = firstLeaf; i < lastLeaf; ++i)
    {
        IBox target = useMixD ? sfcIBox(sfcMixDKey(leaves[i]), sfcMixDKey(leaves[i + 1]), mixDBits.bx, mixDBits.by, mixDBits.bz) : sfcIBox(sfcKey(leaves[i]), sfcKey(leaves[i + 1]));
        if (target.xmax() - target.xmin() == 0 && target.ymax() - target.ymin() == 0 && target.zmax() - target.zmin() == 0)
        {
            continue; // skip empty boxes
        }

        auto violatesMac = [target, ellipse, pbc, &octree, domainStart, domainEnd, mixDBits, useMixD](TreeNodeIndex idx)
        {
            KeyType nodeStart = octree.codeStart(idx);
            KeyType nodeEnd   = octree.codeEnd(idx);
            // if the tree node with index idx is fully contained in the focus, we stop traversal
            if (containedIn(nodeStart, nodeEnd, domainStart, domainEnd)) { return false; }

            IBox source = useMixD ? sfcIBox(sfcMixDKey(nodeStart), maxTreeLevel<KeyType>{} - octree.level(idx), mixDBits.bx, mixDBits.by, mixDBits.bz) : sfcIBox(sfcKey(nodeStart), octree.level(idx));
            if (source.xmax() - source.xmin() == 0 && source.ymax() - source.ymin() == 0 && source.zmax() - source.zmin() == 0)
            {
                return false; // skip empty boxes
            }
            return !minMacMutualInt(target, source, ellipse, pbc);
        };

        auto markLeafIdx = [&octree, &peers, &assignment](TreeNodeIndex idx)
        {
            int peerRank    = assignment.findRank(octree.codeStart(idx));
            peers[peerRank] = 1;
        };

        singleTraversal(octree.childOffsets().data(), octree.parents().data(), violatesMac, markLeafIdx);
    }

    std::vector<int> ret;
    for (int i = 0; i < int(peers.size()); ++i)
    {
        if (peers[i]) { ret.push_back(i); }
    }

    return ret;
}

} // namespace cstone
