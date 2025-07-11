/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Traits and functors for the MPI-enabled FocusedOctree
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <iostream>
#include <numeric>

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/cuda/device_vector.h"
#include "cstone/domain/layout.hpp"
#include "cstone/focus/exchange_focus.hpp"
#include "cstone/focus/octree_focus.hpp"
#include "cstone/focus/source_center.hpp"
#include "cstone/focus/source_center_gpu.h"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/traversal/collisions_gpu.h"

#include <ranges>

namespace cstone
{

//! @brief A fully traversable octree with a local focus
template<class KeyType, class RealType, class Accelerator = CpuTag>
class FocusedOctree
{
    //! @brief A vector template that resides on the hardware specified as Accelerator
    template<class ValueType>
    using AccVector = std::conditional_t<HaveGpu<Accelerator>{}, DeviceVector<ValueType>, std::vector<ValueType>>;

public:
    /*! @brief constructor
     *
     * @param myRank        executing rank id
     * @param numRanks      number of ranks
     * @param bucketSize    Maximum number of particles per leaf inside the focus area
     */
    FocusedOctree(int myRank, int numRanks, unsigned bucketSize)
        : myRank_(myRank)
        , numRanks_(numRanks)
        , bucketSize_(bucketSize)
        , treelets_(numRanks_)
        , macsAcc_(1, 1)
        , centersAcc_(1)
        , globNumNodes_(numRanks)
        , globDispl_(numRanks + 1)
    {
        octreeAcc_.resize(1);
        leaves_ = std::vector<KeyType>{0, nodeRange<KeyType>(0)};
        leafCounts_ = std::vector<unsigned>{bucketSize + 1};
        countsAcc_ = leafCounts_;

        if constexpr (HaveGpu<Accelerator>{})
        {
            leavesAcc_ = leaves_;
            buildOctreeGpu(rawPtr(leavesAcc_), octreeAcc_.data());
            downloadOctree();

            reallocate(geoCentersAcc_, 1, 1.0);
        }
        else { updateInternalTree<KeyType>(leaves_, octreeAcc_.data()); }
    }

    /*! @brief Update the tree structure according to previously calculated criteria (MAC and particle counts)
     *
     * @param[in] peerRanks        list of ranks with nodes that fail the MAC in the SFC part assigned to @p myRank
     * @param[in] assignment       assignment of the global leaf tree to ranks
     * @param[in] box              global coordinate bounding box
     * @param     scratch          memory buffer for temporary usage, on device for the GPU version
     * @return                     true if the tree structure did not change
     *
     * The part of the SFC that is assigned to @p myRank is considered as the focus area.
     */
    template<class Vector>
    bool updateTree(std::span<const int> peerRanks,
                    const SfcAssignment<KeyType>& assignment,
                    const Box<RealType>& box,
                    Vector& scratch)
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

        assert(leafCounts_.size() == octreeAcc_.numLeafNodes);
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
                {rawPtr(countsAcc_), countsAcc_.size()}, {rawPtr(macsAcc_), macsAcc_.size()}, scratch);

            while (not macRefineGpu(octreeAcc_, leavesAcc_, centersAcc_, macsAcc_, prevFocusStart, prevFocusEnd,
                                    focusStart, focusEnd, invThetaRefine, box))
                ;

            reallocateDestructive(leaves_, leavesAcc_.size(), allocGrowthRate_);
            memcpyD2H(rawPtr(leavesAcc_), leavesAcc_.size(), rawPtr(leaves_));
        }
        else
        {
            converged = CombinedUpdate<KeyType>::updateFocus(octreeAcc_, leaves_, bucketSize_, focusStart, focusEnd,
                                                             enforcedKeys, countsAcc_, macsAcc_);
            while (not macRefine(octreeAcc_, leaves_, centersAcc_, macsAcc_, prevFocusStart, prevFocusEnd, focusStart,
                                 focusEnd, invThetaRefine, box))
                ;
        }
        translateAssignment<KeyType>(assignment, leaves_, peers_, myRank_, assignment_);

        if constexpr (HaveGpu<Accelerator>{})
        {
            syncTreeletsGpu<KeyType>(peers_, assignment_, leaves_, octreeAcc_, leavesAcc_, treelets_, scratch);
            downloadOctree();
        }
        else
        {
            syncTreelets(peers_, assignment_, octreeAcc_, leaves_, treelets_);
            hostPrefixes_ = octreeAcc_.prefixes;
        }

        indexTreelets<KeyType>(peerRanks, hostPrefixes_, octreeAcc_.levelRange, treelets_, treeletIdx_);

        translateAssignment<KeyType>(assignment, leaves_, peers_, myRank_, assignment_);
        std::copy_n(assignment.numNodesPerRankConst().begin(), numRanks_, globNumNodes_.begin());
        std::copy_n(assignment.treeOffsetsConst().begin(), numRanks_ + 1, globDispl_.begin());
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
    void updateCounts(std::span<const KeyType> particleKeys,
                      std::span<const KeyType> globalTreeLeaves,
                      std::span<const unsigned> globalCounts,
                      DeviceVector& scratch)
    {
        std::size_t origSize = scratch.size();
        std::span<const KeyType> leaves(leaves_);

        leafCounts_.resize(nNodes(leaves_));
        if constexpr (HaveGpu<Accelerator>{})
        {
            reallocateDestructive(leafCountsAcc_, nNodes(leavesAcc_), allocGrowthRate_);
            TreeNodeIndex numLeafNodes = octreeAcc_.numLeafNodes;

            computeNodeCountsGpu(rawPtr(leavesAcc_), rawPtr(leafCountsAcc_), numLeafNodes, particleKeys,
                                 std::numeric_limits<unsigned>::max(), false);

            // add counts from global tree
            auto idxFromGlob       = enumerateRanges(invertRanges(0, assignment_, numLeafNodes));
            std::size_t numIndices = idxFromGlob.size();
            auto* d_indices        = util::packAllocBuffer<TreeNodeIndex>(scratch, {&numIndices, 1}, 64)[0].data();
            memcpyH2D(idxFromGlob.data(), idxFromGlob.size(), d_indices);

            std::span<const KeyType> leavesAcc{rawPtr(leavesAcc_), leavesAcc_.size()};
            rangeCountGpu<KeyType>(globalTreeLeaves, globalCounts, leavesAcc, {d_indices, idxFromGlob.size()},
                                   {rawPtr(leafCountsAcc_), leafCountsAcc_.size()});

            // 1st upsweep with local and global data
            reallocateDestructive(countsAcc_, octreeAcc_.numNodes, allocGrowthRate_);
            scatterGpu(leafToInternal(octreeAcc_).data(), numLeafNodes, rawPtr(leafCountsAcc_), rawPtr(countsAcc_));

            upsweepSumGpu(maxTreeLevel<KeyType>{}, rawPtr(octreeAcc_.levelRange), rawPtr(octreeAcc_.childOffsets),
                          rawPtr(countsAcc_));
            std::span<unsigned> countsAccView{rawPtr(countsAcc_), countsAcc_.size()};
            peerExchangeGpu(countsAccView, static_cast<int>(P2pTags::focusPeerCounts), scratch);

            upsweepSumGpu(maxTreeLevel<KeyType>{}, rawPtr(octreeAcc_.levelRange), rawPtr(octreeAcc_.childOffsets),
                          rawPtr(countsAcc_));
            gatherAcc<HaveGpu<Accelerator>{}>(leafToInternal(octreeAcc_), rawPtr(countsAcc_), rawPtr(leafCountsAcc_));

            memcpyD2H(rawPtr(leafCountsAcc_), octreeAcc_.numLeafNodes, rawPtr(leafCounts_));
        }
        else
        {
            // local node counts
            assert(std::is_sorted(particleKeys.begin(), particleKeys.end()));
            computeNodeCounts<KeyType>(leaves_.data(), leafCounts_.data(), nNodes(leaves_), particleKeys,
                                       std::numeric_limits<unsigned>::max(), true);

            // add counts from global tree
            auto idxFromGlob = enumerateRanges(invertRanges(0, assignment_, nNodes(leaves_)));
            rangeCount<KeyType>(globalTreeLeaves, globalCounts, leaves, idxFromGlob, leafCounts_);

            // 1st upsweep with local and global data
            countsAcc_.resize(octreeAcc_.numNodes);
            scatter<TreeNodeIndex>(leafToInternal(octreeAcc_), leafCounts_.data(), countsAcc_.data());
            upsweep(octreeAcc_.levelRange, octreeAcc_.childOffsets.data(), countsAcc_.data(), NodeCount<unsigned>{});

            // add counts from neighboring peers
            peerExchange(std::span(countsAcc_), static_cast<int>(P2pTags::focusPeerCounts), scratch);

            // 2nd upsweep with peer data present
            upsweep(octreeAcc_.levelRange, octreeAcc_.childOffsets.data(), countsAcc_.data(), NodeCount<unsigned>{});
            gather(leafToInternal(octreeAcc_), countsAcc_.data(), leafCounts_.data());
        }
        reallocate(scratch, origSize, 1.0);

        rebalanceStatus_ |= countsCriterion;
    }

    template<class T, class DevVec>
    void peerExchange(std::span<T> q, int commTag, DevVec& s) const
    {
        exchangeTreeletGeneral<T>(peers_, treeletIdx_.view(), assignment_, leafToInternal(octreeAcc_), q, commTag, s);
    }

    template<class T, class DevVec>
    void peerExchangeGpu(std::span<T> q, int commTag, DevVec& s) const
    {
        exchangeTreeletGeneral<T>(peers_, treeletIdxAcc_.view(), assignment_, leafToInternal(octreeAcc_), q, commTag,
                                  s);
    }

    /*! @brief transfer quantities of leaf cells inside the focus into a global array
     *
     * @tparam     T                 an arithmetic type or compile-time constant size arrays thereof
     * @param[in]  gLeaves           cstone SFC key leaf cell array of the global tree
     * @param[in]  localQuantities   cell properties of the locally focused tree, length = octree().numTreeNodes()
     * @param[out] globalQuantities  cell properties of the global tree
     * @param[-]   gmapScratch       scratch space for global tree indices to populate, length = gLeaves.size()
     */
    template<class T>
    void populateGlobal(std::span<const KeyType> gLeaves,
                        std::span<const T> localQuantities,
                        std::span<T> globalQuantities,
                        std::span<TreeNodeIndex> gmapScratch) const
    {
        auto gmap       = gmapScratch.subspan(globDispl_[myRank_], globNumNodes_[myRank_]);
        auto gLeavesFoc = gLeaves.subspan(globDispl_[myRank_], globNumNodes_[myRank_] + 1);

        if constexpr (HaveGpu<Accelerator>{})
        {
            locateNodesGpu(gLeavesFoc.data(), gLeavesFoc.data() + gLeavesFoc.size(), octreeAcc_.prefixes.data(),
                           octreeAcc_.d_levelRange.data(), gmap.data());
        }
        else
        {
            const KeyType* nodeKeys         = rawPtr(octreeAcc_.prefixes);
            const TreeNodeIndex* levelRange = rawPtr(octreeAcc_.levelRange);

#pragma omp parallel for schedule(static)
            for (TreeNodeIndex i = 0; i < globNumNodes_[myRank_]; ++i)
            {
                gmap[i] = locateNode(gLeavesFoc[i], gLeavesFoc[i + 1], nodeKeys, levelRange);
            }
        }

        gatherAcc<HaveGpu<Accelerator>{}, TreeNodeIndex>(gmap, localQuantities.data(),
                                                         globalQuantities.data() + globDispl_[myRank_]);
    }

    /*! @brief transfer missing cell quantities from global tree into localQuantities
     *
     * @tparam     T                 an arithmetic type or compile-time constant size arrays thereof
     * @param[in]  globalNodeKeys    WS-prefix-bit keys of the global octree
     * @param[in]  globalLevelRange  tree node index ranges per level for @p globalNodeKeys
     * @param[in]  globalQuantities  tree cell properties for each cell in @p globalTree include internal cells
     * @param[out] localQuantities   local tree cell properties
     * @param[-]   letIdxBuf         temp buffer, size = octreeAcc_.numLeafNodes - assignment_[myRank_].count()
     *                               size is an upper bound, fewer elements may be used
     * @param[-]   letToGlobBuf      temp buffer, size = letIdx.size()
     */
    template<class T>
    void extractGlobal(const KeyType* globalNodeKeys,
                       const TreeNodeIndex* globalLevelRange,
                       std::span<const T> globalQuantities,
                       std::span<T> localQuantities,
                       TreeNodeIndex* letIdxBuf,
                       TreeNodeIndex* letToGlobBuf) const
    {
        //! list of leaf cell indices in the locally focused tree that need global information
        auto idxFromGlob                = enumerateRanges(invertRanges(0, assignment_, octreeAcc_.numLeafNodes));
        const TreeNodeIndex* toInternal = leafToInternal(octreeAcc_).data();
        std::span letIdx{letIdxBuf, idxFromGlob.size()};
        std::span letToGlob{letToGlobBuf, idxFromGlob.size()};
        if constexpr (HaveGpu<Accelerator>{})
        {
            memcpyH2D(idxFromGlob.data(), idxFromGlob.size(), letIdx.data());
            gatherGpu(letIdx.data(), idxFromGlob.size(), toInternal, letIdx.data());

            locateNodesGpu(octreeAcc_.prefixes.data(), letIdx.data(), idxFromGlob.size(), globalNodeKeys,
                           globalLevelRange, letToGlob.data());
            gatherScatterGpu(letToGlob.data(), letIdx.data(), idxFromGlob.size(), globalQuantities.data(),
                             localQuantities.data());
        }
        else
        {
            gather<TreeNodeIndex>(idxFromGlob, toInternal, idxFromGlob.data());
#pragma omp parallel for schedule(static)
            for (TreeNodeIndex i = 0; i < idxFromGlob.size(); ++i)
            {
                letToGlob[i] = locateNode(octreeAcc_.prefixes[idxFromGlob[i]], globalNodeKeys, globalLevelRange);
            }
            gatherScatter<TreeNodeIndex>(letToGlob, idxFromGlob, globalQuantities.data(), localQuantities.data());

        }
    }

    //! @brief Distribute global leaf quantities with local part filled in
    template<class T>
    void gatherGlobalLeaves(std::span<T> globalLeafQuantities) const
    {
        if constexpr (HaveGpu<Accelerator>{}) { syncGpu(); }
        mpiAllgathervGpuDirect<HaveGpu<Accelerator>{}>(MPI_IN_PLACE, 0, globalLeafQuantities.data(),
                                                       globNumNodes_.data(), globDispl_.data(), MPI_COMM_WORLD);
    }

    template<class Tm, class DevVec1 = std::vector<LocalIndex>, class DevVec2 = std::vector<LocalIndex>>
    void updateCenters(const RealType* x,
                       const RealType* y,
                       const RealType* z,
                       const Tm* m,
                       OctreeView<const KeyType> gOctree,
                       DevVec1&& scratch1 = std::vector<LocalIndex>{},
                       DevVec2&& scratch2 = std::vector<LocalIndex>{})
    {
        assert(gOctree.leaves != nullptr);
        TreeNodeIndex firstIdx           = assignment_[myRank_].start();
        TreeNodeIndex lastIdx            = assignment_[myRank_].end();
        OctreeView<const KeyType> octree = octreeViewAcc();

        reallocate(gOctree.numNodes, allocGrowthRate_, globalCentersAcc_);
        reallocate(octree.numNodes, allocGrowthRate_, centersAcc_);

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
            upsweepCentersGpu(maxTreeLevel<KeyType>{}, octreeAcc_.levelRange.data(), octree.childOffsets,
                              rawPtr(centersAcc_));

            reallocate(scratch1, osz1, 1.0);
        }
        else
        {
            //! compute temporary pre-halo exchange particle layout for local particles only
            std::vector<LocalIndex> layout(leafCounts_.size() + 1, 0);
            std::inclusive_scan(leafCounts_.begin() + firstIdx, leafCounts_.begin() + lastIdx,
                                layout.begin() + firstIdx + 1, std::plus<>{}, LocalIndex(0));
#pragma omp parallel for schedule(static)
            for (TreeNodeIndex leafIdx = 0; leafIdx < octreeAcc_.numLeafNodes; ++leafIdx)
            {
                //! prepare local leaf centers
                TreeNodeIndex nodeIdx = octree.leafToInternal[octree.numInternalNodes + leafIdx];
                centersAcc_[nodeIdx]  = massCenter<RealType>(x, y, z, m, layout[leafIdx], layout[leafIdx + 1]);
            }
            //! upsweep with local data in place
            upsweep(octreeAcc_.levelRange, octreeAcc_.childOffsets.data(), centersAcc_.data(),
                    CombineSourceCenter<RealType>{});
        }

        //! global exchange for the top nodes that are bigger than local domains
        auto [gLeafCentersAcc, gmapScratch] =
            util::packAllocBuffer(scratch1, util::TypeList<SourceCenterType<RealType>, TreeNodeIndex>{},
                                  {size_t(gOctree.numLeafNodes), size_t(gOctree.numLeafNodes)}, 128);
        populateGlobal<SourceCenterType<RealType>>(gOctree.leafSpan(), {centersAcc_.data(), centersAcc_.size()},
                                                   gLeafCentersAcc, gmapScratch);
        gatherGlobalLeaves<SourceCenterType<RealType>>(gLeafCentersAcc);

        scatterAcc<HaveGpu<Accelerator>{}>(gOctree.leafToInternalSpan(), gLeafCentersAcc.data(),
                                           globalCentersAcc_.data());
        if constexpr (HaveGpu<Accelerator>{})
        {
            upsweepCentersGpu(maxTreeLevel<KeyType>{}, gOctree.levelRange, gOctree.childOffsets,
                              globalCentersAcc_.data());
        }
        else
        {
            upsweep(gOctree.levelRangeSpan(), gOctree.childOffsets, globalCentersAcc_.data(),
                    CombineSourceCenter<RealType>{});
        }

        std::size_t numLetIdx    = octreeAcc_.numLeafNodes - assignment_[myRank_].count();
        auto [letIdx, letToGlob] = util::packAllocBuffer(scratch1, util::TypeList<TreeNodeIndex, TreeNodeIndex>{},
                                                         {numLetIdx, numLetIdx}, 128);

        extractGlobal<SourceCenterType<RealType>>(
            gOctree.prefixes, gOctree.d_levelRange, {globalCentersAcc_.data(), globalCentersAcc_.size()},
            {centersAcc_.data(), centersAcc_.size()}, letIdx.data(), letToGlob.data());

        if constexpr (HaveGpu<Accelerator>{})
        {
            peerExchangeGpu(std::span{centersAcc_.data(), centersAcc_.size()},
                            static_cast<int>(P2pTags::focusPeerCenters), scratch1);
            upsweepCentersGpu(maxTreeLevel<KeyType>{}, octreeAcc_.levelRange.data(), octree.childOffsets,
                              rawPtr(centersAcc_));
        }
        else
        {
            //! exchange information with peer close to focus
            peerExchange(std::span{centersAcc_.data(), centersAcc_.size()}, static_cast<int>(P2pTags::focusPeerCenters),
                         scratch1);
            //! upsweep with all (leaf) data in place
            upsweep(octree.levelRangeSpan(), octree.childOffsets, centersAcc_.data(), CombineSourceCenter<RealType>{});
        }
    }

    /*! @brief Update the MAC criteria based on a min distance MAC
     *
     * @tparam    T            float or double
     * @param[in] assignment   assignment of the global leaf tree to ranks
     * @param[in] invThetaEff  inverse effective opening angle, 1/theta + 0.5
     */
    void updateMinMac(const SfcAssignment<KeyType>& assignment, float invThetaEff, bool accumulate)
    {
        if constexpr (HaveGpu<Accelerator>{})
        {
            reallocate(centersAcc_, octreeAcc_.numNodes, allocGrowthRate_);
            moveCenters(rawPtr(geoCentersAcc_), octreeAcc_.numNodes, rawPtr(centersAcc_));
        }
        else
        {
            centersAcc_.resize(octreeAcc_.numNodes);
            const KeyType* nodeKeys = octreeAcc_.prefixes.data();

#pragma omp parallel for schedule(static)
            for (TreeNodeIndex i = 0; i < octreeAcc_.numNodes; ++i)
            {
                //! set centers to geometric centers for min dist Mac
                centersAcc_[i] = computeMinMacR2(nodeKeys[i], invThetaEff, box_);
            }
        }

        updateMacs(assignment, invThetaEff, accumulate);
    }

    //! @brief Compute MAC acceptance radius of each cell based on @p invTheta and previously computed expansion centers
    void setMacRadius(float invTheta)
    {
        if constexpr (HaveGpu<Accelerator>{})
        {
            setMacGpu(rawPtr(octreeAcc_.prefixes), octreeAcc_.numNodes, rawPtr(centersAcc_), invTheta, box_);
        }
        else { setMac<RealType, KeyType>(octreeAcc_.prefixes, centersAcc_, invTheta, box_); }
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
    void updateMacs(const SfcAssignment<KeyType>& assignment, float invTheta, bool accumulate)
    {
        if (accumulate && TreeNodeIndex(macsAcc_.size()) != octreeAcc_.numNodes)
        {
            throw std::runtime_error("MAC flags not correctly allocated\n");
        }
        setMacRadius(invTheta);
        reallocate(octreeAcc_.numNodes, allocGrowthRate_, macsAcc_);

        // need to find again assignment start and end indices in focus tree because assignment might have changed
        TreeNodeIndex fAssignStart = findNodeAbove(rawPtr(leaves_), nNodes(leaves_), assignment[myRank_]);
        TreeNodeIndex fAssignEnd   = findNodeAbove(rawPtr(leaves_), nNodes(leaves_), assignment[myRank_ + 1]);

        if constexpr (HaveGpu<Accelerator>{})
        {
            if (not accumulate) { fillGpu(rawPtr(macsAcc_), rawPtr(macsAcc_) + macsAcc_.size(), uint8_t(0)); }
            markMacsGpu(rawPtr(octreeAcc_.prefixes), rawPtr(octreeAcc_.childOffsets), rawPtr(octreeAcc_.parents),
                        rawPtr(centersAcc_), box_, rawPtr(leavesAcc_) + fAssignStart, fAssignEnd - fAssignStart, false,
                        rawPtr(macsAcc_));
        }
        else
        {
            if (not accumulate) { std::fill(rawPtr(macsAcc_), rawPtr(macsAcc_) + macsAcc_.size(), uint8_t(0)); }
            markMacs(rawPtr(octreeAcc_.prefixes), rawPtr(octreeAcc_.childOffsets), rawPtr(octreeAcc_.parents),
                     rawPtr(centersAcc_), box_, rawPtr(leaves_) + fAssignStart, fAssignEnd - fAssignStart, false,
                     rawPtr(macsAcc_));
        }

        rebalanceStatus_ |= macCriterion;
    }

    /*! @brief Discover which cells outside myRank's assignment are halos
     *
     * @param[-]  layout           temporary storage for node count scan
     * @param[in] h                smoothing lengths of locally owned particles
     * @param[in] searchExtFact    increases halo search radius to extend the depth of the ghost layer
     * @param[-]  scratch          host or device buffer for temporary use
     */
    template<class Th, class Vector>
    void discoverHalos(std::span<LocalIndex> layout, const Th* h, float searchExtFact, Vector& scratch, bool accumulate)
    {
        TreeNodeIndex firstNode      = assignment_[myRank_].start();
        TreeNodeIndex lastNode       = assignment_[myRank_].end();
        auto let                     = octreeViewAcc();
        TreeNodeIndex numNodesSearch = lastNode - firstNode;
        TreeNodeIndex numLeafNodes   = let.numLeafNodes;

        if (accumulate && TreeNodeIndex(macsAcc_.size()) != let.numNodes)
        {
            throw std::runtime_error("halo flags not correctly allocated\n");
        }
        reallocate(let.numNodes, allocGrowthRate_, macsAcc_);

        if constexpr (HaveGpu<Accelerator>{})
        {
            size_t radiiBytes = numLeafNodes * sizeof(float);
            size_t origSize   = reallocateBytes(scratch, radiiBytes, allocGrowthRate_);
            auto* d_radii     = reinterpret_cast<float*>(rawPtr(scratch));

            fillGpu(layout.data() + firstNode, layout.data() + firstNode + 1, LocalIndex{0});
            inclusiveScanGpu(leafCountsAcc_.data() + firstNode, leafCountsAcc_.data() + lastNode,
                             layout.data() + firstNode + 1);
            segmentMax(h, layout.data() + firstNode, numNodesSearch, d_radii + firstNode);
            // SPH convention: interaction radius = 2 * h
            scaleGpu(d_radii, d_radii + numLeafNodes, 2.0f * searchExtFact);

            if (not accumulate) { fillGpu(rawPtr(macsAcc_), rawPtr(macsAcc_) + macsAcc_.size(), uint8_t(0)); }
            findHalosGpu(let.prefixes, let.childOffsets, let.parents, leavesAcc_.data(), d_radii, box_, firstNode,
                         lastNode, macsAcc_.data());

            reallocate(scratch, origSize, 1.0);
        }
        else
        {
            layout[0] = 0;
            std::inclusive_scan(leafCounts_.begin() + firstNode, leafCounts_.begin() + lastNode, layout.begin() + 1,
                                std::plus{}, LocalIndex{0});
            std::vector<float> haloRadii(leafCounts_.size(), 0.0f);
#pragma omp parallel for schedule(static)
            for (TreeNodeIndex i = 0; i < numNodesSearch; ++i)
            {
                if (layout[i + 1] > layout[i])
                {
                    // Note factor 2 due to SPH convention: interaction radius = 2 * h
                    haloRadii[i + firstNode] = *std::max_element(h + layout[i], h + layout[i + 1]) * 2 * searchExtFact;
                }
            }
            if (not accumulate) { std::fill(rawPtr(macsAcc_), rawPtr(macsAcc_) + macsAcc_.size(), uint8_t(0)); }
            findHalos(let.prefixes, let.childOffsets, let.parents, leaves_.data(), haloRadii.data(), box_, firstNode,
                      lastNode, macsAcc_.data());
        }
    }

    int computeLayout(std::span<LocalIndex> layoutAcc, std::span<LocalIndex> layout) const
    {
        if constexpr (HaveGpu<Accelerator>{})
        {
            computeNodeLayout<true>({leafCountsAcc_.data(), leafCountsAcc().size()}, {macsAcc_.data(), macsAcc_.size()},
                                    leafToInternal(octreeAcc_), assignment_[myRank_], layoutAcc);
            memcpyD2H(layoutAcc.data(), layoutAcc.size(), layout.data());
        }
        else
        {
            computeNodeLayout<false>({leafCounts_.data(), leafCountsAcc().size()}, {macsAcc_.data(), macsAcc_.size()},
                                     leafToInternal(octreeAcc_), assignment_[myRank_], layout);
        }

        return checkLayout(myRank_, assignment_, layout, treeLeaves());
    }

    //! @brief update until converged with a simple min-distance MAC
    template<class DeviceVector = std::vector<KeyType>>
    void converge(const Box<RealType>& box,
                  std::span<const KeyType> particleKeys,
                  std::span<const int> peers,
                  const SfcAssignment<KeyType>& assignment,
                  std::span<const KeyType> globalTreeLeaves,
                  std::span<const unsigned> globalCounts,
                  float invThetaEff,
                  DeviceVector&& scratch = std::vector<KeyType>{})
    {
        int converged = 0;
        while (converged != numRanks_)
        {
            updateMinMac(assignment, invThetaEff, false);
            converged = updateTree(peers, assignment, box, scratch);
            updateCounts(particleKeys, globalTreeLeaves, globalCounts, scratch);
            updateGeoCenters();
            MPI_Allreduce(MPI_IN_PLACE, &converged, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        }
    }

    //! @brief returns the tree depth
    TreeNodeIndex depth() const { return maxDepth(octreeAcc_.levelRange.data(), octreeAcc_.levelRange.size()); }

    //! @brief the cornerstone leaf cell array
    std::span<const KeyType> treeLeaves() const { return leaves_; }
    //! @brief the assignment of the focus tree leaves to peer ranks
    std::span<const TreeIndexPair> assignment() const { return assignment_; }
    //! @brief Expansion (com) centers of each cell
    std::span<const SourceCenterType<RealType>> expansionCentersAcc() const
    {
        return {rawPtr(centersAcc_), centersAcc_.size()};
    }
    //! @brief Expansion (com) centers of each global cell
    std::span<const SourceCenterType<RealType>> globalExpansionCenters() const
    {
        return {globalCentersAcc_.data(), globalCentersAcc_.size()};
    }

    //! @brief return a view to the octree on the active accelerator
    OctreeView<const KeyType> octreeViewAcc() const { return octreeAcc_.cdata(); }

    //! @brief the cornerstone leaf cell array on the accelerator
    std::span<const KeyType> treeLeavesAcc() const
    {
        if constexpr (HaveGpu<Accelerator>{}) { return {rawPtr(leavesAcc_), leavesAcc_.size()}; }
        else { return leaves_; }
    }

    //! @brief the cornerstone leaf cell particle counts
    std::span<const unsigned> leafCountsAcc() const
    {
        if constexpr (HaveGpu<Accelerator>{}) { return {rawPtr(leafCountsAcc_), leafCountsAcc_.size()}; }
        else { return leafCounts_; }
    }

    //! brief particle counts per focus tree leaf cell
    std::span<const unsigned> countsAcc() const { return {rawPtr(countsAcc_), countsAcc_.size()}; }

    std::span<const Vec3<RealType>> geoCentersAcc() const { return {rawPtr(geoCentersAcc_), geoCentersAcc_.size()}; }
    std::span<const Vec3<RealType>> geoSizesAcc() const { return {rawPtr(geoSizesAcc_), geoSizesAcc_.size()}; }

private:
    //! @brief compute geometrical center and size of each tree cell in terms of x,y,z coordinates
    void updateGeoCenters()
    {
        reallocate(geoCentersAcc_, octreeAcc_.numNodes, allocGrowthRate_);
        reallocate(geoSizesAcc_, octreeAcc_.numNodes, allocGrowthRate_);

        if constexpr (HaveGpu<Accelerator>{})
        {
            computeGeoCentersGpu(rawPtr(octreeAcc_.prefixes), octreeAcc_.numNodes, rawPtr(geoCentersAcc_),
                                 rawPtr(geoSizesAcc_), box_);
        }
        else { nodeFpCenters<KeyType>(octreeAcc_.prefixes, geoCentersAcc_.data(), geoSizesAcc_.data(), box_); }
    }

    void downloadOctree()
    {
        if constexpr (HaveGpu<Accelerator>{})
        {
            TreeNodeIndex numLeafNodes = octreeAcc_.numLeafNodes;
            TreeNodeIndex numNodes     = octreeAcc_.numNodes;

            reallocate(numNodes, allocGrowthRate_, hostPrefixes_);
            memcpyD2H(rawPtr(octreeAcc_.prefixes), numNodes, hostPrefixes_.data());

            reallocateDestructive(leaves_, numLeafNodes + 1, allocGrowthRate_);
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

    std::vector<KeyType> hostPrefixes_;
    OctreeData<KeyType, Accelerator> octreeAcc_;

    //! @brief leaves in cstone format for tree_
    std::vector<KeyType> leaves_;
    AccVector<KeyType> leavesAcc_;

    //! @brief previous iteration focus start
    KeyType prevFocusStart = 0;
    //! @brief previous iteration focus end
    KeyType prevFocusEnd = 0;

    //! @brief particle counts of the focused tree leaves, tree_.treeLeaves()
    std::vector<unsigned> leafCounts_;
    AccVector<unsigned> leafCountsAcc_;
    //! @brief particle counts of the full tree, tree_.octree()
    AccVector<unsigned> countsAcc_;
    //! @brief mac evaluation result relative to focus area (pass or fail)
    AccVector<uint8_t> macsAcc_;
    //! @brief the expansion (com) centers of each cell of tree_.octree()
    AccVector<SourceCenterType<RealType>> centersAcc_;
    //! @brief geometric center and size per cell
    AccVector<Vec3<RealType>> geoCentersAcc_;
    AccVector<Vec3<RealType>> geoSizesAcc_;

    //! @brief we also need to hold on to the expansion centers of the global tree for the multipole upsweep
    AccVector<SourceCenterType<RealType>> globalCentersAcc_;
    //! @brief the assignment of peer ranks to tree_.treeLeaves()
    std::vector<TreeIndexPair> assignment_;
    //! @brief number of global nodes per rank and scan for allgatherv
    std::vector<TreeNodeIndex> globNumNodes_, globDispl_;

    //! @brief the status of the macs_ and counts_ rebalance criteria
    int rebalanceStatus_{valid};
};

/*! @brief exchange data of non-peer (beyond focus) tree cells
 *
 * @tparam        Q                an arithmetic type, or compile-time fix-sized arrays thereof
 * @tparam        T                float or double
 * @tparam        F                function object for octree upsweep
 * @param[in]     gOctree          a global (replicated on all ranks) tree
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
template<class Q, class KeyType, class T, class F, class Accelerator, class Vector, class... UArgs>
void globalFocusExchange(OctreeView<const KeyType> gOctree,
                         const FocusedOctree<KeyType, T, Accelerator>& focusTree,
                         std::span<Q> quantities,
                         Vector& scratch,
                         F&& upsweepFunction,
                         UArgs&&... upsweepArgs)
{
    assert(gOctree.leaves != nullptr);
    static_assert(HaveGpu<Accelerator>{} == IsDeviceVector<Vector>{});
    std::size_t numGlobalLeaves = gOctree.numLeafNodes;
    std::size_t numLetIdx       = focusTree.octreeViewAcc().numLeafNodes;
    auto s                      = scratch.size();
    auto [globalLeafQ, globalQuantities, gmapScratch, letIdx, letToGlob] =
        util::packAllocBuffer(scratch, util::TypeList<Q, Q, TreeNodeIndex, TreeNodeIndex, TreeNodeIndex>{},
                              {numGlobalLeaves, size_t(gOctree.numNodes), numGlobalLeaves, numLetIdx, numLetIdx}, 128);
    focusTree.template populateGlobal<Q>(gOctree.leafSpan(), quantities, globalLeafQ, gmapScratch);

    //! exchange global leaves
    focusTree.template gatherGlobalLeaves<Q>(globalLeafQ);

    scatterAcc<HaveGpu<Accelerator>{}>(gOctree.leafToInternalSpan(), globalLeafQ.data(), globalQuantities.data());
    //! upsweep with the global tree
    upsweepFunction(gOctree.levelRangeSpan(), gOctree.childOffsets, globalQuantities.data(), upsweepArgs...);

    //! from the global tree, extract the part that the executing rank was missing
    focusTree.template extractGlobal<Q>(gOctree.prefixes, gOctree.d_levelRange, globalQuantities, quantities,
                                        letIdx.data(), letToGlob.data());
    reallocate(scratch, s, 1.0);
}

} // namespace cstone
