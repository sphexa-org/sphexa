/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Zurich, 2021 University of Basel
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Implementation of halo discovery and halo exchange
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <numeric>
#include <vector>

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/domain/index_ranges.hpp"
#include "cstone/domain/layout.hpp"
#include "cstone/halos/exchange_halos.hpp"
#ifdef USE_CUDA
#include "cstone/halos/exchange_halos_gpu.cuh"
#endif
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/traversal/collisions.hpp"
#include "cstone/traversal/collisions_gpu.h"
#include "cstone/primitives/accel_switch.hpp"
#include "cstone/util/gsl-lite.hpp"
#include "cstone/util/reallocate.hpp"

namespace cstone
{

namespace detail
{

//! @brief check that only owned particles in [particleStart_:particleEnd_] are sent out as halos
static void checkIndices(const SendList& sendList,
                         [[maybe_unused]] LocalIndex start,
                         [[maybe_unused]] LocalIndex end,
                         [[maybe_unused]] LocalIndex bufferSize)
{
    for (const auto& manifest : sendList)
    {
        for (size_t ri = 0; ri < manifest.nRanges(); ++ri)
        {
            assert(!overlapTwoRanges(LocalIndex{0}, start, manifest.rangeStart(ri), manifest.rangeEnd(ri)));
            assert(!overlapTwoRanges(end, bufferSize, manifest.rangeStart(ri), manifest.rangeEnd(ri)));
        }
    }
}

//! @brief check halo discovery for sanity
template<class KeyType>
int checkHalos(int myRank,
               gsl::span<const TreeIndexPair> focusAssignment,
               gsl::span<const int> haloFlags,
               gsl::span<const KeyType> ftree)
{
    TreeNodeIndex firstAssignedNode = focusAssignment[myRank].start();
    TreeNodeIndex lastAssignedNode  = focusAssignment[myRank].end();

    std::array<TreeNodeIndex, 2> checkRanges[2] = {{0, firstAssignedNode},
                                                   {lastAssignedNode, TreeNodeIndex(haloFlags.size())}};

    int ret = 0;
    for (int range = 0; range < 2; ++range)
    {
#pragma omp parallel for
        for (TreeNodeIndex i = checkRanges[range][0]; i < checkRanges[range][1]; ++i)
        {
            if (haloFlags[i])
            {
                bool peerFound = false;
                for (auto peerRange : focusAssignment)
                {
                    if (peerRange.start() <= i && i < peerRange.end()) { peerFound = true; }
                }
                if (!peerFound)
                {
                    std::cout << "Assignment rank " << myRank << " " << std::oct << ftree[firstAssignedNode] << " - "
                              << ftree[lastAssignedNode] << std::dec << std::endl;
                    std::cout << "Failed node " << i << " " << std::oct << ftree[i] << " - " << ftree[i + 1] << std::dec
                              << std::endl;
                    ret = 1;
                }
            }
        }
    }
    return ret;
}

} // namespace detail

template<class DevVec1, class DevVec2, class... Arrays>
void haloExchangeGpu(int epoch,
                     const SendList& incomingHalos,
                     const SendList& outgoingHalos,
                     DevVec1& sendScratchBuffer,
                     DevVec2& receiveScratchBuffer,
                     Arrays... arrays);

template<class KeyType, class Accelerator>
class Halos
{
public:
    Halos(int myRank)
        : myRank_(myRank)
    {
    }

    /*! @brief Discover which cells outside myRank's assignment are halos
     *
     * @param[in] focusedTree      Fully linked octree, focused on the assignment of the executing rank
     * @param[in] counts           (focus) tree counts
     * @param[in] focusAssignment  Assignment of leaf tree cells to ranks
     * @param[-]  layout           temporary storage for node count scan
     * @param[in] box              Global coordinate bounding box
     * @param[in] h                smoothing lengths of locally owned particles
     * @param[in] searchExtFact    increases halo search radius to extend the depth of the ghost layer
     * @param[-]  scratchBuffer    host or device buffer for temporary use
     */
    template<class T, class Th, class Vector>
    void discover(const KeyType* prefixes,
                  const TreeNodeIndex* childOffsets,
                  const TreeNodeIndex* internalToLeaf,
                  const KeyType* leaves,
                  gsl::span<const unsigned> counts,
                  gsl::span<const TreeIndexPair> focusAssignment,
                  gsl::span<LocalIndex> layout,
                  const Box<T>& box,
                  const Th* h,
                  float searchExtFact,
                  Vector& scratch)
    {
        TreeNodeIndex firstNode      = focusAssignment[myRank_].start();
        TreeNodeIndex lastNode       = focusAssignment[myRank_].end();
        TreeNodeIndex numNodesSearch = lastNode - firstNode;
        TreeNodeIndex numLeafNodes   = counts.size();

        float growthRate = 1.05;
        reallocate(numLeafNodes, growthRate, haloFlags_);

        if constexpr (HaveGpu<Accelerator>{})
        {
            // round up to multiple of 128 such that the radii pointer will be aligned
            size_t flagBytes  = round_up((numLeafNodes + 1) * sizeof(int), 128);
            size_t radiiBytes = numLeafNodes * sizeof(float);
            size_t origSize   = reallocateBytes(scratch, flagBytes + radiiBytes, growthRate);

            auto* d_flags = reinterpret_cast<int*>(rawPtr(scratch));
            auto* d_radii = reinterpret_cast<float*>(rawPtr(scratch)) + flagBytes / sizeof(float);

            fillGpu(layout.data() + firstNode, layout.data() + firstNode + 1, LocalIndex(0));
            inclusiveScanGpu(counts.data() + firstNode, counts.data() + lastNode, layout.data() + firstNode + 1);
            segmentMax(h, layout.data() + firstNode, numNodesSearch, d_radii + firstNode);
            // SPH convention: interaction radius = 2 * h
            scaleGpu(d_radii, d_radii + numLeafNodes, 2.0f * searchExtFact);

            fillGpu(d_flags, d_flags + numLeafNodes, 0);
            findHalosGpu(prefixes, childOffsets, internalToLeaf, leaves, d_radii, box, firstNode, lastNode, d_flags);
            memcpyD2H(d_flags, numLeafNodes, haloFlags_.data());

            reallocate(scratch, origSize, 1.0);
        }
        else
        {
            layout[0] = 0;
            std::inclusive_scan(counts.begin() + firstNode, counts.begin() + lastNode, layout.begin() + 1,
                                std::plus<>{}, LocalIndex(0));
            std::vector<float> haloRadii(counts.size(), 0.0f);
#pragma omp parallel for schedule(static)
            for (TreeNodeIndex i = 0; i < numNodesSearch; ++i)
            {
                if (layout[i + 1] > layout[i])
                {
                    // Note factor 2 due to SPH convention: interaction radius = 2 * h
                    haloRadii[i + firstNode] = *std::max_element(h + layout[i], h + layout[i + 1]) * 2 * searchExtFact;
                }
            }
            std::fill(begin(haloFlags_), end(haloFlags_), 0);
            findHalos(prefixes, childOffsets, internalToLeaf, leaves, haloRadii.data(), box, firstNode, lastNode,
                      haloFlags_.data());
        }
    }

    /*! @brief Compute particle offsets of each tree node and determine halo send/receive indices
     *
     * @param[in]  leaves      (focus) tree leaves
     * @param[in]  counts      (focus) tree counts
     * @param[in]  assignment  assignment of @p leaves to ranks
     * @param[in]  peers       list of peer ranks
     * @param[out] layout      Particle offsets for each node in @p leaves w.r.t to the final particle buffers,
     *                         including the halos, length = counts.size() + 1. The last element contains
     *                         the total number of locally present particles, i.e. assigned + halos.
     *                         [layout[i]:layout[i+1]] indexes particles in the i-th leaf cell.
     *                         If the i-th cell is not a halo and not locally owned, its particles are not present
     *                         and the corresponding layout range has length zero.
     * @return                 0 if all halo cells have been matched with a peer rank, 1 otherwise
     */
    int computeLayout(gsl::span<const KeyType> leaves,
                      gsl::span<const unsigned> counts,
                      gsl::span<const TreeIndexPair> assignment,
                      gsl::span<const int> peers,
                      gsl::span<LocalIndex> layout)
    {
        computeNodeLayout(counts, haloFlags_, assignment[myRank_].start(), assignment[myRank_].end(), layout);
        auto newParticleStart = layout[assignment[myRank_].start()];
        auto newParticleEnd   = layout[assignment[myRank_].end()];

        outgoingHaloIndices_ = exchangeRequestKeys<KeyType>(leaves, haloFlags_, assignment, peers, layout);

        if (detail::checkHalos(myRank_, assignment, haloFlags_, leaves)) { return 1; }
        detail::checkIndices(outgoingHaloIndices_, newParticleStart, newParticleEnd, layout.back());

        incomingHaloIndices_ = computeHaloRecvList(layout, haloFlags_, assignment, peers);
        return 0;
    }

    /*! @brief repeat the halo exchange pattern from the previous sync operation for a different set of arrays
     *
     * @param[inout] arrays  std::vector<float or double> of size particleBufferSize_
     *
     * Arrays are not resized or reallocated. Function is const, but modifies mutable haloEpoch_ counter.
     * Note that if the ScratchVectors are on device, all arrays need to be on the device too.
     */
    template<class Scratch1, class Scratch2, class... Vectors>
    void exchangeHalos(std::tuple<Vectors&...> arrays, Scratch1& sendBuffer, Scratch2& receiveBuffer) const
    {
        if constexpr (HaveGpu<Accelerator>{})
        {
            static_assert(IsDeviceVector<Scratch1>{} && IsDeviceVector<Scratch2>{});
            std::apply(
                [this, &sendBuffer, &receiveBuffer](auto&... arrays)
                {
                    haloExchangeGpu(haloEpoch_++, incomingHaloIndices_, outgoingHaloIndices_, sendBuffer, receiveBuffer,
                                    rawPtr(arrays)...);
                },
                arrays);
        }
        else
        {
            std::apply([this](auto&... arrays)
                       { haloexchange(haloEpoch_++, incomingHaloIndices_, outgoingHaloIndices_, rawPtr(arrays)...); },
                       arrays);
        }
    }

    gsl::span<int> haloFlags() { return haloFlags_; }

private:
    int myRank_;

    RecvList incomingHaloIndices_;
    SendList outgoingHaloIndices_;

    std::vector<int> haloFlags_;

    /*! @brief Counter for halo exchange calls
     * Multiple client calls to domain::exchangeHalos() during a time-step
     * should get different MPI tags, because there is no global MPI_Barrier or MPI collective in between them.
     */
    mutable int haloEpoch_{0};
};

} // namespace cstone
