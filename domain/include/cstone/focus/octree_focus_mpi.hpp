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
 * @brief Traits and functors for the MPI-enabled FocusedOctree
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <numeric>

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/cuda/device_vector.h"
#include "cstone/domain/layout.hpp"
#include "cstone/focus/exchange_focus.hpp"
#include "cstone/focus/octree_focus.hpp"
#include "cstone/focus/source_center.hpp"
#include "cstone/focus/source_center_gpu.h"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/primitives/accel_switch.hpp"
#include "cstone/traversal/collisions_gpu.h"

namespace cstone
{

//! @brief A fully traversable octree with a local focus
template<class KeyType, class RealType, class Accelerator = CpuTag>
class FocusedOctree
{
    //! @brief A vector template that resides on the hardware specified as Accelerator
    template<class ValueType>
    using AccVector = typename AccelSwitchType<Accelerator, std::vector, DeviceVector>::template type<ValueType>;

public:
    /*! @brief constructor
     *
     * @param myRank        executing rank id
     * @param numRanks      number of ranks
     * @param bucketSize    Maximum number of particles per leaf inside the focus area
     * @param theta         Opening angle parameter for a min-distance MAC criterion.
     *                      In a converged FocusedOctree, each node outside the focus area
     *                      passes the min-distance MAC with theta as the parameter w.r.t
     *                      to any point inside the focus area.
     */
    FocusedOctree(int myRank, int numRanks, unsigned bucketSize)
        : myRank_(myRank)
        , numRanks_(numRanks)
        , bucketSize_(bucketSize)
        , treelets_(numRanks_)
        , counts_{bucketSize + 1}
        , macs_{1}
        , centers_(1)
        , globAssignment_(numRanks + 1)
    {
        if constexpr (HaveGpu<Accelerator>{})
        {
            std::vector<KeyType> init{0, nodeRange<KeyType>(0)};
            reallocate(leavesAcc_, init.size(), 1.0);
            memcpyH2D(init.data(), init.size(), rawPtr(leavesAcc_));
            octreeAcc_.resize(nNodes(leavesAcc_));
            buildOctreeGpu(rawPtr(leavesAcc_), octreeAcc_.data());

            reallocate(countsAcc_, counts_.size(), 1.0);
            memcpyH2D(counts_.data(), counts_.size(), rawPtr(countsAcc_));

            reallocate(macsAcc_, macs_.size(), 1.0);
            reallocate(geoCentersAcc_, centers_.size(), 1.0);
            reallocate(centersAcc_, centers_.size(), 1.0);
        }

        leaves_ = std::vector<KeyType>{0, nodeRange<KeyType>(0)};
        treeData_.resize(nNodes(leaves_));
        updateInternalTree<KeyType>(leaves_, treeData_.data());
    }

    /*! @brief Update the tree structure according to previously calculated criteria (MAC and particle counts)
     *
     * @param[in] peerRanks        list of ranks with nodes that fail the MAC in the SFC part assigned to @p myRank
     * @param[in] assignment       assignment of the global leaf tree to ranks
     * @return                     true if the tree structure did not change
     *
     * The part of the SFC that is assigned to @p myRank is considered as the focus area.
     */
    bool updateTree(gsl::span<const int> peerRanks, const SfcAssignment<KeyType>& assignment, const Box<RealType>& box)
    {
        if (rebalanceStatus_ != valid)
        {
            throw std::runtime_error("update of criteria required before updating the tree structure\n");
        }
        peers_.resize(peerRanks.size());
        std::copy(peerRanks.begin(), peerRanks.end(), peers_.begin());

        KeyType focusStart = assignment[myRank_];
        KeyType focusEnd   = assignment[myRank_ + 1];
        // init on first call
        if (prevFocusStart == 0 && prevFocusEnd == 0)
        {
            prevFocusStart = focusStart;
            prevFocusEnd   = focusEnd;
        }

        std::vector<KeyType> enforcedKeys;
        enforcedKeys.reserve(peers_.size() * 2);

        focusTransfer<KeyType>(leaves_, leafCounts_, bucketSize_, myRank_, prevFocusStart, prevFocusEnd, focusStart,
                               focusEnd, enforcedKeys);
        for (int peer : peers_)
        {
            enforcedKeys.push_back(assignment[peer]);
            enforcedKeys.push_back(assignment[peer + 1]);
        }
        auto uniqueEnd = std::unique(enforcedKeys.begin(), enforcedKeys.end());
        enforcedKeys.erase(uniqueEnd, enforcedKeys.end());

        float invThetaRefine = sqrt(3) / 2 + 1e-6;
        bool converged;
        if constexpr (HaveGpu<Accelerator>{})
        {
            converged = CombinedUpdate<KeyType>::updateFocusGpu(
                octreeAcc_, leavesAcc_, bucketSize_, focusStart, focusEnd, enforcedKeys,
                {rawPtr(countsAcc_), countsAcc_.size()}, {rawPtr(macsAcc_), macsAcc_.size()});

            while (not macRefineGpu(octreeAcc_, leavesAcc_, centersAcc_, macsAcc_, prevFocusStart, prevFocusEnd,
                                    focusStart, focusEnd, invThetaRefine, box))
                ;

            reallocateDestructive(leaves_, leavesAcc_.size(), allocGrowthRate_);
            memcpyD2H(rawPtr(leavesAcc_), leavesAcc_.size(), rawPtr(leaves_));
        }
        else
        {
            converged = CombinedUpdate<KeyType>::updateFocus(treeData_, leaves_, bucketSize_, focusStart, focusEnd,
                                                             enforcedKeys, counts_, macs_);
            while (not macRefine(treeData_, leaves_, centers_, macs_, prevFocusStart, prevFocusEnd, focusStart,
                                 focusEnd, invThetaRefine, box))
                ;
        }
        translateAssignment<KeyType>(assignment, leaves_, peers_, myRank_, assignment_);

        if constexpr (HaveGpu<Accelerator>{})
        {
            syncTreeletsGpu<KeyType>(peers_, assignment_, leaves_, octreeAcc_, leavesAcc_, treelets_);
            downloadOctree();
        }
        else { syncTreelets(peers_, assignment_, treeData_, leaves_, treelets_); }

        indexTreelets<KeyType>(peerRanks, treeData_.prefixes, treeData_.levelRange, treelets_, treeletIdx_);

        translateAssignment<KeyType>(assignment, leaves_, peers_, myRank_, assignment_);
        std::copy_n(assignment.data(), numRanks_ + 1, globAssignment_.data());
        copy(treeletIdx_, treeletIdxAcc_);

        /*! Store box for use in all property updates (counts, centers, MACs, etc) until updateTree() is called again.
         *  We store it here in order to disallow calling updateMacs with a changed bounding box, because changing
         *  the bounding box invalidates the expansion centers (centersAcc_)
         */
        box_             = box;
        prevFocusStart   = focusStart;
        prevFocusEnd     = focusEnd;
        rebalanceStatus_ = invalid;
        updateGeoCenters();
        return converged;
    }

    /*! @brief Perform a global update of the tree structure
     *
     * @param[in] particleKeys     SFC keys of local particles
     * @param[in] globalTreeLeaves global cornerstone leaf tree
     * @param[in] globalCounts     global cornerstone leaf tree counts
     * @return                     true if the tree structure did not change
     *
     * The part of the SFC that is assigned to @p myRank is considered as the focus area.
     *
     * Preconditions:
     *  - The provided assignment and globalTreeLeaves are the same as what was used for
     *    calculating the list of peer ranks with findPeersMac. (not checked)
     *  - All local particle keys must lie within the assignment of @p myRank (checked)
     *    and must be sorted in ascending order (checked)
     */
    template<class DeviceVector = std::vector<KeyType>>
    void updateCounts(gsl::span<const KeyType> particleKeys,
                      gsl::span<const KeyType> globalTreeLeaves,
                      gsl::span<const unsigned> globalCounts,
                      DeviceVector& scratch)
    {
        std::size_t origSize = scratch.size();
        gsl::span<const KeyType> leaves(leaves_);

        leafCounts_.resize(nNodes(leaves_));
        if constexpr (HaveGpu<Accelerator>{})
        {
            reallocateDestructive(leafCountsAcc_, nNodes(leavesAcc_), allocGrowthRate_);
            TreeNodeIndex numLeafNodes = octreeAcc_.numLeafNodes;

            computeNodeCountsGpu(rawPtr(leavesAcc_), rawPtr(leafCountsAcc_), numLeafNodes, particleKeys.begin(),
                                 particleKeys.end(), std::numeric_limits<unsigned>::max(), false);

            // add counts from global tree
            auto idxFromGlob       = enumerateRanges(invertRanges(0, assignment_, numLeafNodes));
            std::size_t numIndices = idxFromGlob.size();
            auto* d_indices        = util::packAllocBuffer<TreeNodeIndex>(scratch, {&numIndices, 1}, 64)[0].data();
            memcpyH2D(idxFromGlob.data(), idxFromGlob.size(), d_indices);

            gsl::span<const KeyType> leavesAcc{rawPtr(leavesAcc_), leavesAcc_.size()};
            rangeCountGpu<KeyType>(globalTreeLeaves, globalCounts, leavesAcc, {d_indices, idxFromGlob.size()},
                                   {rawPtr(leafCountsAcc_), leafCountsAcc_.size()});

            // 1st upsweep with local and global data
            reallocateDestructive(countsAcc_, octreeAcc_.numNodes, allocGrowthRate_);
            scatterGpu(leafToInternal(octreeAcc_).data(), numLeafNodes, rawPtr(leafCountsAcc_), rawPtr(countsAcc_));

            upsweepSumGpu(maxTreeLevel<KeyType>{}, rawPtr(treeData_.levelRange), rawPtr(octreeAcc_.childOffsets),
                          rawPtr(countsAcc_));
            gsl::span<unsigned> countsAccView{rawPtr(countsAcc_), countsAcc_.size()};
            peerExchangeGpu(countsAccView, static_cast<int>(P2pTags::focusPeerCounts), scratch);

            upsweepSumGpu(maxTreeLevel<KeyType>{}, rawPtr(treeData_.levelRange), rawPtr(octreeAcc_.childOffsets),
                          rawPtr(countsAcc_));
            gatherAcc<HaveGpu<Accelerator>{}>(leafToInternal(octreeAcc_), rawPtr(countsAcc_), rawPtr(leafCountsAcc_));

            memcpyD2H(rawPtr(leafCountsAcc_), octreeAcc_.numLeafNodes, rawPtr(leafCounts_));
        }
        else
        {
            // local node counts
            assert(std::is_sorted(particleKeys.begin(), particleKeys.end()));
            computeNodeCounts(leaves_.data(), leafCounts_.data(), nNodes(leaves_), particleKeys.data(),
                              particleKeys.data() + particleKeys.size(), std::numeric_limits<unsigned>::max(), true);

            // add counts from global tree
            auto idxFromGlob = enumerateRanges(invertRanges(0, assignment_, nNodes(leaves_)));
            rangeCount<KeyType>(globalTreeLeaves, globalCounts, leaves, idxFromGlob, leafCounts_);

            // 1st upsweep with local and global data
            counts_.resize(treeData_.numNodes);
            scatter<TreeNodeIndex>(leafToInternal(treeData_), leafCounts_.data(), counts_.data());
            upsweep(treeData_.levelRange, treeData_.childOffsets, counts_.data(), NodeCount<unsigned>{});

            // add counts from neighboring peers
            peerExchange(gsl::span<unsigned>(counts_), static_cast<int>(P2pTags::focusPeerCounts), scratch);

            // 2nd upsweep with peer data present
            upsweep(treeData_.levelRange, treeData_.childOffsets, counts_.data(), NodeCount<unsigned>{});
            gather(leafToInternal(treeData_), counts_.data(), leafCounts_.data());
        }
        reallocate(scratch, origSize, 1.0);

        rebalanceStatus_ |= countsCriterion;
    }

    template<class T, class DevVec>
    void peerExchange(gsl::span<T> q, int commTag, DevVec& s) const
    {
        exchangeTreeletGeneral<T>(peers_, treeletIdx_.view(), assignment_, leafToInternal(treeData_), q, commTag, s);
    }

    template<class T, class DevVec>
    void peerExchangeGpu(gsl::span<T> q, int commTag, DevVec& s) const
    {
        exchangeTreeletGeneral<T>(peers_, treeletIdxAcc_.view(), assignment_, leafToInternal(octreeAcc_), q, commTag,
                                  s);
    }

    /*! @brief transfer quantities of leaf cells inside the focus into a global array
     *
     * @tparam     T                 an arithmetic type or compile-time constant size arrays thereof
     * @param[in]  globalLeaves      cstone SFC key leaf cell array of the global tree
     * @param[in]  localQuantities   cell properties of the locally focused tree, length = octree().numTreeNodes()
     * @param[out] globalQuantities  cell properties of the global tree
     */
    template<class T>
    void populateGlobal(gsl::span<const KeyType> globalLeaves,
                        gsl::span<const T> localQuantities,
                        gsl::span<T> globalQuantities) const
    {
        assert(localQuantities.size() == treeData_.numNodes);

        TreeNodeIndex firstGlobIdx = findNodeAbove(globalLeaves.data(), globalLeaves.size(), prevFocusStart);
        TreeNodeIndex lastGlobIdx  = findNodeAbove(globalLeaves.data(), globalLeaves.size(), prevFocusEnd);
        auto globLeavesFoc         = globalLeaves.subspan(firstGlobIdx, lastGlobIdx - firstGlobIdx + 1);
        auto globQFoc              = globalQuantities.subspan(firstGlobIdx, lastGlobIdx - firstGlobIdx);

        const KeyType* nodeKeys         = rawPtr(treeData_.prefixes);
        const TreeNodeIndex* levelRange = rawPtr(treeData_.levelRange);

        std::vector<TreeNodeIndex> gmap(lastGlobIdx - firstGlobIdx);
#pragma omp parallel for schedule(static)
        for (TreeNodeIndex i = 0; i < globLeavesFoc.size() - 1; ++i)
        {
            gmap[i] = locateNode(globLeavesFoc[i], globLeavesFoc[i + 1], nodeKeys, levelRange);
        }
        gather<TreeNodeIndex>(gmap, localQuantities.data(), globQFoc.data());
    }

    /*! @brief transfer missing cell quantities from global tree into localQuantities
     *
     * @tparam     T                 an arithmetic type or compile-time constant size arrays thereof
     * @param[in]  globalTree
     * @param[in]  globalQuantities  tree cell properties for each cell in @p globalTree include internal cells
     * @param[out] localQuantities   local tree cell properties
     */
    template<class T>
    void extractGlobal(const KeyType* globalNodeKeys,
                       const TreeNodeIndex* globalLevelRange,
                       gsl::span<const T> globalQuantities,
                       gsl::span<T> localQuantities) const
    {
        const KeyType* prefixes         = rawPtr(treeData_.prefixes);
        const TreeNodeIndex* toInternal = leafToInternal(treeData_).data();
        //! list of leaf cell indices in the locally focused tree that need global information
        auto idxFromGlob = enumerateRanges(invertRanges(0, assignment_, treeData_.numLeafNodes));
        std::transform(idxFromGlob.begin(), idxFromGlob.end(), idxFromGlob.begin(),
                       [m = toInternal](auto i) { return m[i]; });

        std::vector<TreeNodeIndex> gmap(idxFromGlob.size());
#pragma omp parallel for schedule(static)
        for (TreeNodeIndex i = 0; i < idxFromGlob.size(); ++i)
        {
            gmap[i] = locateNode(prefixes[idxFromGlob[i]], globalNodeKeys, globalLevelRange);
        }
        gatherScatter<TreeNodeIndex>(gmap, idxFromGlob, globalQuantities.data(), localQuantities.data());
    }

    //! @brief Distribute global leaf quantities with local part filled in
    template<class T>
    void gatherGlobalLeaves(gsl::span<const KeyType> globalLeaves, gsl::span<T> globalLeafQuantities) const
    {
        std::vector<TreeNodeIndex> numGlobNodesPerRank(numRanks_), globNodesDispl(numRanks_ + 1);

        for (int rank = 0; rank < numRanks_; ++rank)
        {
            globNodesDispl[rank] = findNodeAbove(globalLeaves.data(), nNodes(globalLeaves), globAssignment_[rank]);
        }
        globNodesDispl.back() = nNodes(globalLeaves);
        for (int rank = 0; rank < numRanks_; ++rank)
        {
            numGlobNodesPerRank[rank] = globNodesDispl[rank + 1] - globNodesDispl[rank];
        }

        mpiAllgatherv(MPI_IN_PLACE, 0, globalLeafQuantities.data(), numGlobNodesPerRank.data(), globNodesDispl.data(),
                      MPI_COMM_WORLD);
    }

    template<class Tm, class DevVec1 = std::vector<LocalIndex>, class DevVec2 = std::vector<LocalIndex>>
    void updateCenters(const RealType* x,
                       const RealType* y,
                       const RealType* z,
                       const Tm* m,
                       const Octree<KeyType>& globalTree,
                       DevVec1&& scratch1 = std::vector<LocalIndex>{},
                       DevVec2&& scratch2 = std::vector<LocalIndex>{})
    {
        TreeNodeIndex firstIdx           = assignment_[myRank_].start();
        TreeNodeIndex lastIdx            = assignment_[myRank_].end();
        OctreeView<const KeyType> octree = octreeViewAcc();
        TreeNodeIndex numNodes           = octree.numInternalNodes + octree.numLeafNodes;

        globalCenters_.resize(globalTree.numTreeNodes());
        centers_.resize(numNodes);
        reallocateDestructive(centersAcc_, centers_.size(), allocGrowthRate_);

        if constexpr (HaveGpu<Accelerator>{})
        {
            static_assert(IsDeviceVector<std::decay_t<DevVec1>>{} && IsDeviceVector<std::decay_t<DevVec2>>{});
            size_t bytesLayout = (octree.numLeafNodes + 1) * sizeof(LocalIndex);
            size_t osz1        = reallocateBytes(scratch1, bytesLayout, allocGrowthRate_);
            auto* d_layout     = reinterpret_cast<LocalIndex*>(rawPtr(scratch1));

            fillGpu(d_layout, d_layout + octree.numLeafNodes + 1, LocalIndex(0));
            inclusiveScanGpu(rawPtr(leafCountsAcc_) + firstIdx, rawPtr(leafCountsAcc_) + lastIdx,
                             d_layout + firstIdx + 1);
            computeLeafSourceCenterGpu(x, y, z, m, octree.leafToInternal + octree.numInternalNodes, octree.numLeafNodes,
                                       d_layout, rawPtr(centersAcc_));
            //! upsweep with local data in place
            upsweepCentersGpu(maxTreeLevel<KeyType>{}, treeData_.levelRange.data(), octree.childOffsets,
                              rawPtr(centersAcc_));
            memcpyD2H(rawPtr(centersAcc_), numNodes, centers_.data());

            reallocate(scratch1, osz1, 1.0);
        }
        else
        {
            //! compute temporary pre-halo exchange particle layout for local particles only
            std::vector<LocalIndex> layout(leafCounts_.size() + 1, 0);
            std::inclusive_scan(leafCounts_.begin() + firstIdx, leafCounts_.begin() + lastIdx,
                                layout.begin() + firstIdx + 1, std::plus<>{}, LocalIndex(0));
#pragma omp parallel for schedule(static)
            for (TreeNodeIndex leafIdx = 0; leafIdx < treeData_.numLeafNodes; ++leafIdx)
            {
                //! prepare local leaf centers
                TreeNodeIndex nodeIdx = octree.leafToInternal[octree.numInternalNodes + leafIdx];
                centers_[nodeIdx]     = massCenter<RealType>(x, y, z, m, layout[leafIdx], layout[leafIdx + 1]);
            }
            //! upsweep with local data in place
            upsweep(treeData_.levelRange, treeData_.childOffsets, centers_.data(), CombineSourceCenter<RealType>{});
        }

        //! global exchange for the top nodes that are bigger than local domains
        std::vector<SourceCenterType<RealType>> globalLeafCenters(globalTree.numLeafNodes());
        populateGlobal<SourceCenterType<RealType>>(globalTree.treeLeaves(), centers_, globalLeafCenters);
        gatherGlobalLeaves<SourceCenterType<RealType>>(globalTree.treeLeaves(), globalLeafCenters);
        scatter(globalTree.internalOrder(), globalLeafCenters.data(), globalCenters_.data());
        upsweep(globalTree.levelRange(), globalTree.childOffsets(), globalCenters_.data(),
                CombineSourceCenter<RealType>{});
        extractGlobal<SourceCenterType<RealType>>(globalTree.nodeKeys().data(), globalTree.levelRange().data(),
                                                  globalCenters_, centers_);

        if constexpr (HaveGpu<Accelerator>{})
        {
            reallocate(centersAcc_, octreeAcc_.numNodes, allocGrowthRate_);
            memcpyH2D(centers_.data(), centers_.size(), rawPtr(centersAcc_));
            peerExchangeGpu<SourceCenterType<RealType>>(centersAcc_, static_cast<int>(P2pTags::focusPeerCenters),
                                                        scratch1);
            upsweepCentersGpu(maxTreeLevel<KeyType>{}, treeData_.levelRange.data(), octree.childOffsets,
                              rawPtr(centersAcc_));
        }
        else
        {
            //! exchange information with peer close to focus
            std::vector<int, util::DefaultInitAdaptor<int>> hScratch;
            peerExchange<SourceCenterType<RealType>>(centers_, static_cast<int>(P2pTags::focusPeerCenters), hScratch);
            //! upsweep with all (leaf) data in place
            upsweep(treeData_.levelRange, treeData_.childOffsets, centers_.data(), CombineSourceCenter<RealType>{});
        }
    }

    /*! @brief Update the MAC criteria based on a min distance MAC
     *
     * @tparam    T            float or double
     * @param[in] assignment   assignment of the global leaf tree to ranks
     * @param[in] invTheta     inverse effective opening angle, 1/theta + 0.5
     */
    void updateMinMac(const SfcAssignment<KeyType>& assignment, float invThetaEff)
    {
        if constexpr (HaveGpu<Accelerator>{})
        {
            reallocate(centersAcc_, octreeAcc_.numNodes, allocGrowthRate_);
            moveCenters(rawPtr(geoCentersAcc_), octreeAcc_.numNodes, rawPtr(centersAcc_));
        }
        else
        {
            centers_.resize(treeData_.numNodes);
            const KeyType* nodeKeys = treeData_.prefixes.data();

#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < treeData_.numNodes; ++i)
            {
                //! set centers to geometric centers for min dist Mac
                centers_[i] = computeMinMacR2(nodeKeys[i], invThetaEff, box_);
            }
        }

        updateMacs(assignment, invThetaEff);
    }

    //! @brief Compute MAC acceptance radius of each cell based on @p invTheta and previously computed expansion centers
    void setMacRadius(float invTheta)
    {
        if constexpr (HaveGpu<Accelerator>{})
        {
            setMacGpu(rawPtr(octreeAcc_.prefixes), octreeAcc_.numNodes, rawPtr(centersAcc_), invTheta, box_);
        }
        else { setMac<RealType, KeyType>(treeData_.prefixes, centers_, invTheta, box_); }
    }

    /*! @brief Update the MAC criteria based on given expansion centers and effective inverse theta
     *
     * @param[in] assignment   assignment of the global leaf tree to ranks
     * @param[in] invTheta     inverse effective opening angle, 1/theta + x
     *
     * Inputs per tree cell:  centers_/centersAcc_  ->  Outputs per tree cell:  macs_/macsAcc_
     *
     * MAC accepted if d > l * invTheta + ||center - geocenter||
     * Based on the provided expansion centers and values of invTheta, different MAC criteria can be implemented:
     *
     * centers_ = center of mass, invTheta = 1/theta        -> "Vector MAC"
     * centers_ = geo centers, invTheta = 1/theta + sqrt(3) -> Worst case vector MAC with center of mass in cell corner
     * centers_ = geo centers, invTheta = 1/theta + 0.5     -> Identical to MinMac along the axes through the center,
     *                                                         slightly less restrictive in the diagonal directions
     */
    void updateMacs(const SfcAssignment<KeyType>& assignment, float invTheta)
    {
        setMacRadius(invTheta);
        macs_.resize(treeData_.numNodes);

        // need to find again assignment start and end indices in focus tree because assignment might have changed
        TreeNodeIndex fAssignStart = findNodeAbove(rawPtr(leaves_), nNodes(leaves_), assignment[myRank_]);
        TreeNodeIndex fAssignEnd   = findNodeAbove(rawPtr(leaves_), nNodes(leaves_), assignment[myRank_ + 1]);

        if constexpr (HaveGpu<Accelerator>{})
        {
            reallocate(macsAcc_, octreeAcc_.numNodes, allocGrowthRate_);
            fillGpu(rawPtr(macsAcc_), rawPtr(macsAcc_) + macsAcc_.size(), char(0));
            markMacsGpu(rawPtr(octreeAcc_.prefixes), rawPtr(octreeAcc_.childOffsets), rawPtr(centersAcc_), box_,
                        rawPtr(leavesAcc_) + fAssignStart, fAssignEnd - fAssignStart, false, rawPtr(macsAcc_));

            memcpyD2H(rawPtr(macsAcc_), macsAcc_.size(), macs_.data());
        }
        else
        {
            std::fill(rawPtr(macs_), rawPtr(macs_) + macs_.size(), char(0));
            markMacs(rawPtr(treeData_.prefixes), rawPtr(treeData_.childOffsets), rawPtr(centers_), box_,
                     rawPtr(leaves_) + fAssignStart, fAssignEnd - fAssignStart, false, rawPtr(macs_));
        }

        rebalanceStatus_ |= macCriterion;
    }

    //! @brief update until converged with a simple min-distance MAC
    template<class DeviceVector = std::vector<KeyType>>
    void converge(const Box<RealType>& box,
                  gsl::span<const KeyType> particleKeys,
                  gsl::span<const int> peers,
                  const SfcAssignment<KeyType>& assignment,
                  gsl::span<const KeyType> globalTreeLeaves,
                  gsl::span<const unsigned> globalCounts,
                  float invThetaEff,
                  DeviceVector&& scratch = std::vector<KeyType>{})
    {
        int converged = 0;
        while (converged != numRanks_)
        {
            updateMinMac(assignment, invThetaEff);
            converged = updateTree(peers, assignment, box);
            updateCounts(particleKeys, globalTreeLeaves, globalCounts, scratch);
            updateGeoCenters();
            MPI_Allreduce(MPI_IN_PLACE, &converged, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        }
    }

    //! @brief returns the tree depth
    TreeNodeIndex depth() const { return maxDepth(treeData_.levelRange.data(), treeData_.levelRange.size()); }

    //! @brief the cornerstone leaf cell array
    gsl::span<const KeyType> treeLeaves() const { return leaves_; }
    //! @brief the assignment of the focus tree leaves to peer ranks
    gsl::span<const TreeIndexPair> assignment() const { return assignment_; }
    //! @brief Expansion (com) centers of each cell
    gsl::span<const SourceCenterType<RealType>> expansionCentersAcc() const
    {
        if constexpr (HaveGpu<Accelerator>{}) { return {rawPtr(centersAcc_), centersAcc_.size()}; }
        else { return centers_; }
    }
    //! @brief Expansion (com) centers of each global cell
    gsl::span<const SourceCenterType<RealType>> globalExpansionCenters() const { return globalCenters_; }

    //! @brief return a view to the octree on the active accelerator
    OctreeView<const KeyType> octreeViewAcc() const
    {
        if constexpr (HaveGpu<Accelerator>{}) { return ((const decltype(octreeAcc_)&)octreeAcc_).data(); }
        else { return treeData_.data(); }
    }

    //! @brief the cornerstone leaf cell array on the accelerator
    gsl::span<const KeyType> treeLeavesAcc() const
    {
        if constexpr (HaveGpu<Accelerator>{}) { return {rawPtr(leavesAcc_), leavesAcc_.size()}; }
        else { return leaves_; }
    }

    //! brief particle counts per focus tree leaf cell
    gsl::span<const unsigned> leafCounts() const { return leafCounts_; }

    //! @brief the cornerstone leaf cell particle counts
    gsl::span<const unsigned> leafCountsAcc() const
    {
        if constexpr (HaveGpu<Accelerator>{}) { return {rawPtr(leafCountsAcc_), leafCountsAcc_.size()}; }
        else { return leafCounts_; }
    }

    //! brief particle counts per focus tree leaf cell
    gsl::span<const unsigned> countsAcc() const { return {rawPtr(countsAcc_), countsAcc_.size()}; }

    gsl::span<const Vec3<RealType>> geoCentersAcc() const { return {rawPtr(geoCentersAcc_), geoCentersAcc_.size()}; }
    gsl::span<const Vec3<RealType>> geoSizesAcc() const { return {rawPtr(geoSizesAcc_), geoSizesAcc_.size()}; }

    void addMacs(gsl::span<int> haloFlags) const
    {
        const TreeNodeIndex* toInternal = leafToInternal(treeData_).data();
#pragma omp parallel for schedule(static)
        for (TreeNodeIndex i = 0; i < haloFlags.ssize(); ++i)
        {
            size_t iIdx = toInternal[i];
            if (macs_[iIdx] && !haloFlags[i]) { haloFlags[i] = 1; }
        }
    }

private:
    //! @brief compute geometrical center and size of each tree cell in terms of x,y,z coordinates
    void updateGeoCenters()
    {
        reallocate(geoCentersAcc_, treeData_.numNodes, allocGrowthRate_);
        reallocate(geoSizesAcc_, treeData_.numNodes, allocGrowthRate_);

        if constexpr (HaveGpu<Accelerator>{})
        {
            computeGeoCentersGpu(rawPtr(octreeAcc_.prefixes), treeData_.numNodes, rawPtr(geoCentersAcc_),
                                 rawPtr(geoSizesAcc_), box_);
        }
        else { nodeFpCenters<KeyType>(treeData_.prefixes, geoCentersAcc_.data(), geoSizesAcc_.data(), box_); }
    }

    void uploadOctree()
    {
        if constexpr (HaveGpu<Accelerator>{})
        {
            TreeNodeIndex numLeafNodes = treeData_.numLeafNodes;
            TreeNodeIndex numNodes     = treeData_.numNodes;

            octreeAcc_.resize(numLeafNodes);
            reallocateDestructive(leavesAcc_, numLeafNodes + 1, allocGrowthRate_);

            memcpyH2D(treeData_.prefixes.data(), numNodes, rawPtr(octreeAcc_.prefixes));
            memcpyH2D(treeData_.childOffsets.data(), numNodes, rawPtr(octreeAcc_.childOffsets));
            memcpyH2D(treeData_.parents.data(), treeData_.parents.size(), rawPtr(octreeAcc_.parents));
            memcpyH2D(treeData_.levelRange.data(), treeData_.levelRange.size(), rawPtr(octreeAcc_.levelRange));
            memcpyH2D(treeData_.internalToLeaf.data(), numNodes, rawPtr(octreeAcc_.internalToLeaf));
            memcpyH2D(treeData_.leafToInternal.data(), numNodes, rawPtr(octreeAcc_.leafToInternal));

            memcpyH2D(leaves_.data(), numLeafNodes + 1, rawPtr(leavesAcc_));
        }
    }

    void downloadOctree()
    {
        if constexpr (HaveGpu<Accelerator>{})
        {
            TreeNodeIndex numLeafNodes = octreeAcc_.numLeafNodes;
            TreeNodeIndex numNodes     = octreeAcc_.numNodes;

            treeData_.resize(numLeafNodes);
            reallocateDestructive(leaves_, numLeafNodes + 1, allocGrowthRate_);

            memcpyD2H(rawPtr(octreeAcc_.prefixes), numNodes, treeData_.prefixes.data());
            memcpyD2H(rawPtr(octreeAcc_.childOffsets), numNodes, treeData_.childOffsets.data());
            memcpyD2H(rawPtr(octreeAcc_.parents), octreeAcc_.parents.size(), treeData_.parents.data());
            memcpyD2H(rawPtr(octreeAcc_.levelRange), octreeAcc_.levelRange.size(), treeData_.levelRange.data());
            memcpyD2H(rawPtr(octreeAcc_.internalToLeaf), numNodes, treeData_.internalToLeaf.data());
            memcpyD2H(rawPtr(octreeAcc_.leafToInternal), numNodes, treeData_.leafToInternal.data());

            memcpyD2H(rawPtr(leavesAcc_), numLeafNodes + 1, leaves_.data());
        }
    }

    enum Status : int
    {
        invalid         = 0,
        countsCriterion = 1,
        macCriterion    = 2,
        // the status is valid for rebalancing if both the counts and macs have been updated
        // since the last call to updateTree
        valid = countsCriterion | macCriterion
    };

    //! @brief the executing rank
    int myRank_;
    //! @brief the total number of ranks
    int numRanks_;
    //! @brief bucket size (ncrit) inside the focus are
    unsigned bucketSize_;

    //! @brief allocation growth rate for focus tree arrays with length ~ numFocusNodes
    float allocGrowthRate_{1.05};
    //! @brief box from last call to updateTree()
    Box<RealType> box_{0, 1};

    //! @brief list of peer ranks from last call to updateTree()
    std::vector<int> peers_;
    //! @brief the tree structures that the peers have for the domain of the executing rank (myRank_)
    std::vector<std::vector<KeyType>> treelets_;
    ConcatVector<TreeNodeIndex> treeletIdx_;
    ConcatVector<TreeNodeIndex, AccVector> treeletIdxAcc_;

    //! @brief octree data resident on GPU if active
    OctreeData<KeyType, Accelerator> octreeAcc_;
    AccVector<KeyType> leavesAcc_;
    AccVector<unsigned> leafCountsAcc_;
    AccVector<SourceCenterType<RealType>> centersAcc_;
    AccVector<Vec3<RealType>> geoCentersAcc_;
    AccVector<Vec3<RealType>> geoSizesAcc_;
    AccVector<unsigned> countsAcc_;
    AccVector<char> macsAcc_;

    OctreeData<KeyType, CpuTag> treeData_;
    //! @brief leaves in cstone format for tree_
    std::vector<KeyType> leaves_;

    //! @brief previous iteration focus start
    KeyType prevFocusStart = 0;
    //! @brief previous iteration focus end
    KeyType prevFocusEnd = 0;

    //! @brief particle counts of the focused tree leaves, tree_.treeLeaves()
    std::vector<unsigned> leafCounts_;
    //! @brief particle counts of the full tree, tree_.octree()
    std::vector<unsigned> counts_;
    //! @brief mac evaluation result relative to focus area (pass or fail)
    std::vector<char> macs_;
    //! @brief the expansion (com) centers of each cell of tree_.octree()
    std::vector<SourceCenterType<RealType>> centers_;
    //! @brief we also need to hold on to the expansion centers of the global tree for the multipole upsweep
    std::vector<SourceCenterType<RealType>> globalCenters_;
    //! @brief the assignment of peer ranks to tree_.treeLeaves()
    std::vector<TreeIndexPair> assignment_;
    //! @brief global domain boundary SFC keys
    std::vector<KeyType> globAssignment_;

    //! @brief the status of the macs_ and counts_ rebalance criteria
    int rebalanceStatus_{valid};
};

/*! @brief exchange data of non-peer (beyond focus) tree cells
 *
 * @tparam        Q                an arithmetic type, or compile-time fix-sized arrays thereof
 * @tparam        T                float or double
 * @tparam        F                function object for octree upsweep
 * @param[in]     globalOctree     a global (replicated on all ranks) tree
 * @param[in]     focusTree        octree focused on the executing rank
 * @param[inout]  quantities       an array of length focusTree.octree().numTreeNodes() with cell properties of the
 *                                 locally focused octree
 * @param[in]     upsweepFunction  callable object that will be used to compute internal cell properties of the
 *                                 global tree based on global leaf quantities
 * @param[in]     upsweepArgs      additional arguments that might be required for a tree upsweep, such as expansion
 *                                 centers if Q is a multipole type.
 *
 * This function obtains missing information for tree cell quantities belonging to far-away ranks which are not
 * peer ranks of the executing rank.
 *
 * The data flow is:
 * cell quantities owned by executing rank -> globalLeafQuantities -> global collective communication -> upsweep
 *   -> back-contribution from globalQuantities into @p quantities
 *
 * Precondition:  quantities contains valid data for each cell, including internal cells,
 *                that fall into the focus range of the executing rank
 * Postcondition: each element of quantities corresponding to non-local cells not owned by any of the peer
 *                ranks contains data obtained through global collective communication between ranks
 */
template<class Q, class KeyType, class T, class F, class Accelerator, class... UArgs>
void globalFocusExchange(const Octree<KeyType>& globalOctree,
                         const FocusedOctree<KeyType, T, Accelerator>& focusTree,
                         gsl::span<Q> quantities,
                         F&& upsweepFunction,
                         UArgs&&... upsweepArgs)
{
    TreeNodeIndex numGlobalLeaves = globalOctree.numLeafNodes();
    std::vector<Q> globalLeafQuantities(numGlobalLeaves);
    focusTree.template populateGlobal<Q>(globalOctree.treeLeaves(), quantities, globalLeafQuantities);

    //! exchange global leaves
    focusTree.template gatherGlobalLeaves<Q>(globalOctree.treeLeaves(), globalLeafQuantities);

    std::vector<Q> globalQuantities(globalOctree.numTreeNodes());
    scatter(globalOctree.internalOrder(), globalLeafQuantities.data(), globalQuantities.data());
    //! upsweep with the global tree
    upsweepFunction(globalOctree.levelRange(), globalOctree.childOffsets(), globalQuantities.data(), upsweepArgs...);

    //! from the global tree, extract the part that the executing rank was missing
    focusTree.template extractGlobal<Q>(globalOctree.nodeKeys().data(), globalOctree.levelRange().data(),
                                        globalQuantities, quantities);
}

} // namespace cstone
