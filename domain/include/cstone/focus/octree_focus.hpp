/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
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
 * @brief Generation of locally essential global octrees in cornerstone format
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * A locally essential octree has a certain global resolution specified by a maximum
 * particle count per leaf node. In addition, it features a focus area defined as a
 * sub-range of the global space filling curve. In this focus sub-range, the resolution
 * can be higher, expressed through a smaller maximum particle count per leaf node.
 * Crucially, the resolution is also higher in the halo-areas of the focus sub-range.
 * These halo-areas can be defined as the overlap with the smoothing-length spheres around
 * the contained particles in the focus sub-range (SPH) or as the nodes whose opening angle
 * is too big to satisfy a multipole acceptance criterion from any perspective within the
 * focus sub-range (N-body).
 */

#pragma once

#include <vector>

#include "cstone/domain/index_ranges.hpp"
#include "cstone/focus/inject.hpp"
#include "cstone/focus/rebalance.hpp"
#include "cstone/focus/rebalance_gpu.h"
#include "cstone/focus/source_center.hpp"
#include "cstone/focus/source_center_gpu.h"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/traversal/collisions_gpu.h"
#include "cstone/traversal/macs.hpp"
#include "cstone/tree/csarray_gpu.h"
#include "cstone/tree/octree.hpp"
#include "cstone/tree/octree_gpu.h"
#include "cstone/traversal/traversal.hpp"
#include "cstone/util/gsl-lite.hpp"

namespace cstone
{

//! @brief Encapsulation to allow making this a friend of Octree<KeyType>
template<class KeyType>
struct CombinedUpdate
{
    /*! @brief combined update of a tree based on count-bucketsize in the focus and based on macs outside
     *
     * @param[inout] tree         the fully linked octree
     * @param[in] bucketSize      maximum node particle count inside the focus area
     * @param[in] focusStart      start of the focus area
     * @param[in] focusEnd        end of the focus area
     * @param[in] mandatoryKeys   List of SFC keys that have to be present in the focus tree after this function
     *                            returns. @p focusStart and @p focusEnd are always mandatory, so they don't need to be
     *                            specified here. @p mandatoryKeys need not be sorted and can tolerate duplicates.
     *                            This is used e.g. to guarantee that the assignment boundaries of peer ranks are
     *                            resolved, even if the update did not converge.
     * @param[in] counts          node particle counts (including internal nodes), length = tree_.numTreeNodes()
     * @param[in] macs            MAC pass/fail results for each node, length = tree_.numTreeNodes()
     * @return                    true if the tree structure did not change
     */
    static bool updateFocus(OctreeData<KeyType, CpuTag>& tree,
                            std::vector<KeyType>& leaves,
                            unsigned bucketSize,
                            KeyType focusStart,
                            KeyType focusEnd,
                            gsl::span<const KeyType> mandatoryKeys,
                            gsl::span<const unsigned> counts,
                            gsl::span<const char> macs)
    {
        [[maybe_unused]] TreeNodeIndex numNodes = tree.numLeafNodes + tree.numInternalNodes;
        assert(TreeNodeIndex(counts.size()) == numNodes);
        assert(TreeNodeIndex(macs.size()) == numNodes);
        assert(TreeNodeIndex(tree.internalToLeaf.size()) >= numNodes);

        // take op decision per node
        gsl::span<TreeNodeIndex> nodeOpsAll(tree.internalToLeaf);
        rebalanceDecisionEssential<KeyType>(tree.prefixes, tree.childOffsets.data(), tree.parents.data(), counts.data(),
                                            macs.data(), focusStart, focusEnd, bucketSize, nodeOpsAll.data());

        std::vector<KeyType> allMandatoryKeys{focusStart, focusEnd};
        std::copy(mandatoryKeys.begin(), mandatoryKeys.end(), std::back_inserter(allMandatoryKeys));
        auto status = enforceKeys<KeyType>(allMandatoryKeys, tree.prefixes.data(), tree.childOffsets.data(),
                                           tree.parents.data(), nodeOpsAll.data());

        bool converged = protectAncestors<KeyType>(tree.prefixes, tree.parents.data(), nodeOpsAll.data());

        // extract leaf decision, using childOffsets at temp storage, require +1 for exclusive scan last element
        assert(tree.childOffsets.size() >= size_t(tree.numLeafNodes + 1));
        gsl::span<TreeNodeIndex> nodeOps(tree.childOffsets.data(), tree.numLeafNodes + 1);
        gather(leafToInternal(tree), nodeOpsAll.data(), nodeOps.data());

        if (status == ResolutionStatus::cancelMerge)
        {
            converged = std::all_of(nodeOps.begin(), nodeOps.end() - 1, [](TreeNodeIndex i) { return i == 1; });
        }
        else if (status == ResolutionStatus::rebalance) { converged = false; }

        // carry out rebalance based on nodeOps
        auto& newLeaves = tree.prefixes;
        rebalanceTree(leaves, newLeaves, nodeOps.data());

        // if rebalancing couldn't introduce the mandatory keys, we force-inject them now into the tree
        if (status == ResolutionStatus::failed)
        {
            converged = false;
            injectKeys<KeyType>(newLeaves, allMandatoryKeys);
        }

        swap(newLeaves, leaves);
        tree.resize(nNodes(leaves));
        updateInternalTree<KeyType>(leaves, tree.data());

        return converged;
    }

    /*! @brief combined update of a tree based on count-bucketsize in the focus and based on macs outside
     *
     * @param[inout] tree         the fully linked octree
     * @param[inout] leaves       cornerstone leaf cell array for @p tree
     * @param[in] bucketSize      maximum node particle count inside the focus area
     * @param[in] focusStart      start of the focus area
     * @param[in] focusEnd        end of the focus area
     * @param[in] mandatoryKeys   List of SFC keys that have to be present in the focus tree after this function
     *                            returns. @p focusStart and @p focusEnd are always mandatory, so they don't need to be
     *                            specified here. @p mandatoryKeys need not be sorted and can tolerate duplicates.
     *                            This is used e.g. to guarantee that the assignment boundaries of peer ranks are
     *                            resolved, even if the update did not converge.
     * @param[in] counts          node particle counts (including internal nodes), length = tree_.numTreeNodes()
     * @param[in] macs            MAC pass/fail results for each node, length = tree_.numTreeNodes()
     * @return                    true if the tree structure did not change
     */
    static bool updateFocusGpu(OctreeData<KeyType, GpuTag>& tree,
                               DeviceVector<KeyType>& leaves,
                               unsigned bucketSize,
                               KeyType focusStart,
                               KeyType focusEnd,
                               gsl::span<const KeyType> mandatoryKeys,
                               gsl::span<const unsigned> counts,
                               gsl::span<const char> macs)
    {
        TreeNodeIndex numNodes = tree.numLeafNodes + tree.numInternalNodes;
        assert(TreeNodeIndex(counts.size()) == numNodes);
        assert(TreeNodeIndex(macs.size()) == numNodes);
        assert(TreeNodeIndex(tree.internalToLeaf.size()) >= numNodes);

        // take op decision per node
        gsl::span<TreeNodeIndex> nodeOpsAll(rawPtr(tree.internalToLeaf), numNodes);
        rebalanceDecisionEssentialGpu(rawPtr(tree.prefixes), rawPtr(tree.childOffsets), rawPtr(tree.parents),
                                      counts.data(), macs.data(), focusStart, focusEnd, bucketSize, nodeOpsAll.data(),
                                      numNodes);

        auto status = ResolutionStatus::converged;

        DeviceVector<KeyType> d_mandatoryKeys;
        reallocate(d_mandatoryKeys, mandatoryKeys.size(), 1.0);
        memcpyH2D(mandatoryKeys.data(), mandatoryKeys.size(), rawPtr(d_mandatoryKeys));
        status         = enforceKeysGpu(rawPtr(d_mandatoryKeys), d_mandatoryKeys.size(), rawPtr(tree.prefixes),
                                        rawPtr(tree.childOffsets), rawPtr(tree.parents), nodeOpsAll.data());
        bool converged = protectAncestorsGpu(rawPtr(tree.prefixes), rawPtr(tree.parents), nodeOpsAll.data(), numNodes);

        // extract leaf decision, using childOffsets as temp storage
        assert(tree.childOffsets.size() >= size_t(tree.numLeafNodes + 1));
        gsl::span<TreeNodeIndex> nodeOps(rawPtr(tree.childOffsets), tree.numLeafNodes + 1);
        gatherGpu(leafToInternal(tree).data(), nNodes(leaves), nodeOpsAll.data(), nodeOps.data());

        if (status == ResolutionStatus::cancelMerge)
        {
            converged = countGpu(nodeOps.begin(), nodeOps.end() - 1, 1) == tree.numLeafNodes;
        }
        else if (status == ResolutionStatus::rebalance) { converged = false; }

        exclusiveScanGpu(nodeOps.data(), nodeOps.data() + nodeOps.size(), nodeOps.data());
        TreeNodeIndex newNumLeafNodes;
        memcpyD2H(nodeOps.data() + nodeOps.size() - 1, 1, &newNumLeafNodes);

        auto& newLeaves = tree.prefixes;
        reallocateDestructive(newLeaves, newNumLeafNodes + 1, 1.05);
        rebalanceTreeGpu(rawPtr(leaves), nNodes(leaves), newNumLeafNodes, nodeOps.data(), rawPtr(newLeaves));
        swap(newLeaves, leaves);

        // if rebalancing couldn't introduce the mandatory keys, we force-inject them now into the tree
        if (status == ResolutionStatus::failed)
        {
            converged = false;
            injectKeysGpu(leaves, {d_mandatoryKeys.data(), d_mandatoryKeys.size()}, tree.prefixes, tree.childOffsets,
                          tree.internalToLeaf);
        }

        tree.resize(nNodes(leaves));
        buildOctreeGpu(rawPtr(leaves), tree.data());

        return converged;
    }
};

template<class KeyType>
bool updateMacRefine(OctreeData<KeyType, CpuTag>& tree,
                     std::vector<KeyType>& leaves,
                     gsl::span<const char> macs,
                     TreeIndexPair focus)
{
    assert(tree.childOffsets.size() >= size_t(tree.numLeafNodes + 1));
    gsl::span<TreeNodeIndex> nodeOps(tree.childOffsets.data(), tree.numLeafNodes + 1);

    auto l2i = leafToInternal(tree);
#pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < tree.numLeafNodes; ++i)
    {
        if (i < focus.start() || i >= focus.end()) { nodeOps[i] = macRefineOp(tree.prefixes[l2i[i]], macs[l2i[i]]); }
        else { nodeOps[i] = 1; }
    }

    bool converged  = std::all_of(nodeOps.begin(), nodeOps.end() - 1, [](TreeNodeIndex i) { return i == 1; });
    auto& newLeaves = tree.prefixes;
    rebalanceTree(leaves, newLeaves, nodeOps.data());

    swap(newLeaves, leaves);
    tree.resize(nNodes(leaves));
    updateInternalTree<KeyType>(leaves, tree.data());

    return converged;
}

template<class T, class KeyType>
bool macRefine(OctreeData<KeyType, CpuTag>& tree,
               std::vector<KeyType>& leaves,
               std::vector<SourceCenterType<T>>& centers,
               std::vector<char>& macs,
               KeyType oldFocusStart,
               KeyType oldFocusEnd,
               KeyType focusStart,
               KeyType focusEnd,
               float invTheta,
               const Box<T>& box)
{
    if (oldFocusStart == focusStart && oldFocusEnd == focusEnd) { return true; }
    centers.resize(tree.numNodes);
    geoMacSpheres<KeyType>(tree.prefixes, rawPtr(centers), invTheta, box);

    macs.resize(tree.numNodes);
    std::fill(macs.begin(), macs.end(), 0);

    KeyType growthLower = focusStart < oldFocusStart ? oldFocusStart : focusStart;
    KeyType growthUpper = oldFocusEnd < focusEnd ? oldFocusEnd : focusEnd;

    TreeNodeIndex fGrowL = findNodeAbove(rawPtr(leaves), nNodes(leaves), growthLower);
    TreeNodeIndex fGrowU = findNodeAbove(rawPtr(leaves), nNodes(leaves), growthUpper);
    TreeNodeIndex fStart = findNodeAbove(rawPtr(leaves), nNodes(leaves), focusStart);
    TreeNodeIndex fEnd   = findNodeAbove(rawPtr(leaves), nNodes(leaves), focusEnd);

    markMacs(rawPtr(tree.prefixes), rawPtr(tree.childOffsets), rawPtr(centers), box, rawPtr(leaves) + fStart,
             fGrowL - fStart, true, macs.data());
    markMacs(rawPtr(tree.prefixes), rawPtr(tree.childOffsets), rawPtr(centers), box, rawPtr(leaves) + fGrowU,
             fEnd - fGrowU, true, macs.data());

    return updateMacRefine(tree, leaves, macs, {fStart, fEnd});
}

template<class KeyType>
bool updateMacRefineGpu(OctreeData<KeyType, GpuTag>& tree,
                        DeviceVector<KeyType>& leaves,
                        const char* macs,
                        TreeIndexPair focus)
{
    assert(tree.childOffsets.size() >= size_t(tree.numLeafNodes + 1));
    gsl::span<TreeNodeIndex> nodeOps(rawPtr(tree.childOffsets), tree.numLeafNodes + 1);

    auto l2i = leafToInternal(tree);
    macRefineDecisionGpu(rawPtr(tree.prefixes), macs, l2i.data(), l2i.size(), focus, nodeOps.data());

    bool converged = countGpu(nodeOps.cbegin(), nodeOps.cend() - 1, 1) == tree.numLeafNodes;
    exclusiveScanGpu(nodeOps.data(), nodeOps.data() + nodeOps.size(), nodeOps.data());
    TreeNodeIndex newNumLeafNodes;
    memcpyD2H(nodeOps.data() + nodeOps.size() - 1, 1, &newNumLeafNodes);

    auto& newLeaves = tree.prefixes;
    reallocateDestructive(newLeaves, newNumLeafNodes + 1, 1.05);
    rebalanceTreeGpu(rawPtr(leaves), nNodes(leaves), newNumLeafNodes, nodeOps.data(), rawPtr(newLeaves));
    swap(newLeaves, leaves);

    tree.resize(nNodes(leaves));
    buildOctreeGpu(rawPtr(leaves), tree.data());

    return converged;
}

template<class T, class KeyType>
bool macRefineGpu(OctreeData<KeyType, GpuTag>& tree,
                  DeviceVector<KeyType>& leaves,
                  DeviceVector<SourceCenterType<T>>& centers,
                  DeviceVector<char>& macs,
                  KeyType oldFocusStart,
                  KeyType oldFocusEnd,
                  KeyType focusStart,
                  KeyType focusEnd,
                  float invTheta,
                  const Box<T>& box)
{
    if (oldFocusStart == focusStart && oldFocusEnd == focusEnd) { return true; }
    reallocate(centers, tree.numNodes, 1.05);
    geoMacSpheresGpu(rawPtr(tree.prefixes), tree.numNodes, rawPtr(centers), invTheta, box);

    reallocate(macs, tree.numNodes, 1.05);
    fillGpu(macs.data(), macs.data() + macs.size(), char(0));

    KeyType growthLower = focusStart < oldFocusStart ? oldFocusStart : focusStart;
    KeyType growthUpper = oldFocusEnd < focusEnd ? oldFocusEnd : focusEnd;

    TreeNodeIndex fGrowL = lowerBoundGpu(rawPtr(leaves), rawPtr(leaves) + nNodes(leaves), growthLower);
    TreeNodeIndex fStart = lowerBoundGpu(rawPtr(leaves), rawPtr(leaves) + nNodes(leaves), focusStart);
    TreeNodeIndex fEnd   = lowerBoundGpu(rawPtr(leaves), rawPtr(leaves) + nNodes(leaves), focusEnd);
    TreeNodeIndex fGrowU = lowerBoundGpu(rawPtr(leaves), rawPtr(leaves) + nNodes(leaves), growthUpper);

    markMacsGpu(rawPtr(tree.prefixes), rawPtr(tree.childOffsets), rawPtr(centers), box, rawPtr(leaves) + fStart,
                fGrowL - fStart, true, macs.data());
    markMacsGpu(rawPtr(tree.prefixes), rawPtr(tree.childOffsets), rawPtr(centers), box, rawPtr(leaves) + fGrowU,
                fEnd - fGrowU, true, macs.data());

    return updateMacRefineGpu(tree, leaves, macs.data(), {fStart, fEnd});
}

/*! @brief A fully traversable octree, locally focused w.r.t a MinMac criterion
 *
 * This single rank version is only useful in unit tests.
 */
template<class KeyType>
class FocusedOctreeSingleNode
{
    using CB = CombinedUpdate<KeyType>;

public:
    FocusedOctreeSingleNode(unsigned bucketSize, float theta)
        : theta_(theta)
        , bucketSize_(bucketSize)
        , counts_{bucketSize + 1}
        , macs_{1}
    {
        leaves_ = std::vector<KeyType>{0, nodeRange<KeyType>(0)};
        tree_.resize(nNodes(leaves_));
        updateInternalTree<KeyType>(leaves_, tree_.data());
    }

    //! @brief perform a local update step, see FocusedOctreeCore
    template<class T>
    bool update(const Box<T>& box,
                gsl::span<const KeyType> particleKeys,
                KeyType focusStart,
                KeyType focusEnd,
                gsl::span<const KeyType> mandatoryKeys)
    {
        bool converged =
            CB::updateFocus(tree_, leaves_, bucketSize_, focusStart, focusEnd, mandatoryKeys, counts_, macs_);

        std::vector<Vec4<T>> centers_(tree_.numNodes);
        float invThetaEff = 1.0f / theta_ + 0.5;

#pragma omp parallel for schedule(static)
        for (TreeNodeIndex i = 0; i < tree_.numNodes; ++i)
        {
            //! set centers to geometric centers for min dist Mac
            centers_[i] = computeMinMacR2(tree_.prefixes[i], invThetaEff, box);
        }

        macs_.resize(tree_.numNodes);
        std::fill(macs_.begin(), macs_.end(), 0);
        TreeNodeIndex fStart = findNodeAbove(rawPtr(leaves_), nNodes(leaves_), focusStart);
        TreeNodeIndex fEnd   = findNodeAbove(rawPtr(leaves_), nNodes(leaves_), focusEnd);
        markMacs(tree_.prefixes.data(), tree_.childOffsets.data(), centers_.data(), box, rawPtr(leaves_) + fStart,
                 fEnd - fStart, false, macs_.data());

        leafCounts_.resize(nNodes(leaves_));
        computeNodeCounts(leaves_.data(), leafCounts_.data(), nNodes(leaves_), particleKeys.data(),
                          particleKeys.data() + particleKeys.size(), std::numeric_limits<unsigned>::max(), true);

        counts_.resize(tree_.numNodes);
        scatter(leafToInternal(tree_), leafCounts_.data(), counts_.data());
        upsweep(tree_.levelRange, tree_.childOffsets, counts_.data(), NodeCount<unsigned>{});

        return converged;
    }

    gsl::span<const KeyType> treeLeaves() const { return leaves_; }
    gsl::span<const unsigned> leafCounts() const { return leafCounts_; }

private:
    //! @brief opening angle refinement criterion
    float theta_;
    unsigned bucketSize_;

    OctreeData<KeyType, CpuTag> tree_;
    std::vector<KeyType> leaves_;

    //! @brief particle counts of the focused tree leaves
    std::vector<unsigned> leafCounts_;
    std::vector<unsigned> counts_;
    //! @brief mac evaluation result relative to focus area (pass or fail)
    std::vector<char> macs_;
};

} // namespace cstone
