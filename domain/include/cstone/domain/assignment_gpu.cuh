/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Zurich, 2021 University of Basel
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Implementation of global particle assignment and distribution
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <vector>

#include "cstone/cuda/device_vector.h"
#include "cstone/domain/domaindecomp_gpu.cuh"
#include "cstone/domain/domaindecomp_mpi_gpu.cuh"
#include "cstone/domain/layout.hpp"
#include "cstone/primitives/gather.cuh"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/tree/octree.hpp"
#include "cstone/tree/update_mpi.hpp"
#include "cstone/tree/update_mpi_gpu.cuh"
#include "cstone/sfc/box_mpi.hpp"
#include "cstone/sfc/sfc_gpu.h"

namespace cstone
{

/*! @brief A class for global domain assignment and distribution
 *
 * @tparam KeyType  32- or 64-bit unsigned integer
 * @tparam T        float or double
 *
 * This class holds a low-res global octree which is replicated across all ranks and it presides over
 * the assignment of that tree to the ranks and performs the necessary point-2-point data exchanges
 * to send all particles to their owning ranks.
 */
template<class KeyType, class T>
class GlobalAssignmentGpu
{
public:
    GlobalAssignmentGpu(int rank, int nRanks, unsigned bucketSize, const Box<T>& box)
        : myRank_(rank)
        , numRanks_(nRanks)
        , bucketSize_(bucketSize)
        , box_(box)
    {
        unsigned level            = log8ceil<KeyType>(100 * nRanks);
        auto initialBoundaries    = initialDomainSplits<KeyType>(nRanks, level);
        std::vector<KeyType> init = computeSpanningTree<KeyType>(initialBoundaries);
        tree_.update(init.data(), nNodes(init));
        nodeCounts_ = std::vector<unsigned>(nNodes(init), bucketSize_ - 1);

        reallocate(d_boundaryKeys_, numRanks_ + 1, 1.0);
        reallocate(d_boundaryIndices_, numRanks_ + 1, 1.0);
    }

    /*! @brief Update the global tree
     *
     * @param[in]  bufDesc         Buffer description of @a keys, @a x, @a y, @a z with range of assigned particles
     * @param[in]  sfcSorter       records the SFC order of the owned input coordinates
     * @param[out] particleKeys    will contain sorted particle SFC keys, length = bufDesc.size, on DEVICE
     * @param[-]   s0              device scratch vector
     * @param[-]   s1              device scratch vector
     * @param[in]  x               x coordinates, length = bufDesc.size, ON DEVICE
     * @param[in]  y               y coordinates, length = bufDesc.size, ON DEVICE
     * @param[in]  z               z coordinates, length = bufDesc.size, ON DEVICE
     * @return                     required buffer size for the next call to @a distribute
     *
     * This function does not modify / communicate any particle data.
     */
    template<class Sorter, class DevVector>
    LocalIndex assign(BufferDescription bufDesc,
                      Sorter& sfcSorter,
                      DevVector& s0,
                      DevVector& s1,
                      KeyType* particleKeys,
                      const T* x,
                      const T* y,
                      const T* z)
    {
        // number of locally assigned particles to consider for global tree building
        LocalIndex numParticles = bufDesc.end - bufDesc.start;
        LocalIndex start        = bufDesc.start;

        auto fittingBox = makeGlobalBox<T, MinMaxGpu<T>>(x + start, y + start, z + start, numParticles, box_);
        if (firstCall_) { box_ = fittingBox; }
        else { box_ = limitBoxShrinking(fittingBox, box_); }

        // compute SFC particle keys only for particles participating in tree build
        gsl::span<KeyType> keyView(particleKeys + start, numParticles);
        computeSfcKeysGpu(x + start, y + start, z + start, sfcKindPointer(keyView.data()), numParticles, box_);

        // sort keys and keep track of ordering for later use
        sfcSorter.setMapFromCodes(keyView.begin(), keyView.end(), s0, s1);

        updateOctreeGlobalGpu(keyView.begin(), keyView.end(), bucketSize_, tree_, d_csTree_, nodeCounts_,
                              d_nodeCounts_);
        if (firstCall_)
        {
            firstCall_ = false;
            while (!updateOctreeGlobalGpu(keyView.begin(), keyView.end(), bucketSize_, tree_, d_csTree_, nodeCounts_,
                                          d_nodeCounts_))
                ;
        }

        auto newAssignment = makeSfcAssignment(numRanks_, nodeCounts_, tree_.treeLeaves().data());
        limitBoundaryShifts<KeyType>(assignment_, newAssignment, tree_.treeLeaves(), nodeCounts_);
        assignment_ = std::move(newAssignment);

        exchanges_ = createSendRangesGpu<KeyType>(assignment_, {particleKeys + bufDesc.start, numParticles},
                                                  rawPtr(d_boundaryKeys_), rawPtr(d_boundaryIndices_));

        return domain_exchange::exchangeBufferSize(bufDesc, numPresent(), numAssigned());
    }

    /*! @brief Distribute particles to their assigned ranks based on previous assignment
     *
     * @param[in]    bufDesc            Buffer description with range of assigned particles and total buffer size
     * @param[inout] sfcSorter          contains the ordering that accesses the range [bufDesc.start:bufDesc.end]
     *                                  in SFC order
     * @param[-]     sendScratch        scratch space for send buffers
     * @param[-]     receiveScratch     scratch space for receive buffers
     * @param[in]    keys               particle SFC keys, sorted in [bufDesc.start:bufDesc.end]
     * @param[inout] x                  particle x-coordinates
     * @param[inout] y                  particle y-coordinates
     * @param[inout] z                  particle z-coordinates
     * @param[inout] particleProperties remaining particle properties, h, m, etc.
     * @return                          index denoting the index range start of particles post-exchange
     *                                  plus a span with a view of the assigned particle keys
     *
     * Note: Instead of reordering the particle buffers right here after the exchange, we only keep track
     * of the reorder map that is required to transform the particle buffers into SFC-order. This allows us
     * to defer the reordering until we have done halo discovery. At that time, we know the final location
     * where to put the assigned particles inside the buffer, such that we can reorder directly to the final
     * location. This saves us from having to move around data inside the buffers for a second time.
     */
    template<class Sorter, class DevVector, class... Arrays>
    auto distribute(BufferDescription bufDesc,
                    Sorter& sfcSorter,
                    DevVector& sendScratch,
                    DevVector& receiveScratch,
                    KeyType* keys,
                    T* x,
                    T* y,
                    T* z,
                    Arrays... particleProperties) const
    {
        recvLog_.clear();
        exchangeParticlesGpu(0, recvLog_, exchanges_, myRank_, bufDesc, numAssigned(), sendScratch, receiveScratch,
                             sfcSorter.getMap(), x, y, z, particleProperties...);

        auto [newStart, newEnd]    = domain_exchange::assignedEnvelope(bufDesc, numPresent(), numAssigned());
        LocalIndex envelopeSize    = newEnd - newStart;
        gsl::span<KeyType> keyView = gsl::span<KeyType>(keys + newStart, envelopeSize);

        auto recvStart = domain_exchange::receiveStart(bufDesc, numPresent(), numAssigned());
        auto numRecv   = numAssigned() - numPresent();

        computeSfcKeysGpu(x + recvStart, y + recvStart, z + recvStart, sfcKindPointer(keys + recvStart), numRecv, box_);
        std::make_signed_t<LocalIndex> shifts = -numRecv;
        if (newEnd > bufDesc.end) { shifts = numRecv; }
        sfcSorter.extendMap(shifts, sendScratch);

        // sort keys and keep track of the ordering
        sfcSorter.updateMap(keyView.begin(), keyView.end(), sendScratch, receiveScratch);

        return std::make_tuple(newStart, keyView.subspan(numSendDown(), numAssigned()));
    }

    //! @brief repeat exchange from last call to assign() for arrays on the CPU
    template<class SVec, class... Arrays>
    auto redoExchange(BufferDescription bufDesc,
                      const LocalIndex* ordering,
                      SVec& /*s1*/,
                      SVec& /*s2*/,
                      Arrays... particleProperties) const
    {
        exchangeParticles(1, recvLog_, exchanges_, myRank_, bufDesc, numAssigned(), ordering, particleProperties...);
    }

    //! @brief read only visibility of the global octree leaves to the outside
    gsl::span<const KeyType> treeLeaves() const { return {rawPtr(d_csTree_), d_csTree_.size()}; }
    //! @brief the octree, including the internal part
    const Octree<KeyType>& octree() const { return tree_; }
    //! @brief read only visibility of the global octree leaf counts to the outside
    gsl::span<const unsigned> nodeCounts() const { return {rawPtr(d_nodeCounts_), d_nodeCounts_.size()}; }
    //! @brief the global coordinate bounding box
    const Box<T>& box() const { return box_; }
    //! @brief return the space filling curve rank assignment of the last call to @a assign()
    const SfcAssignment<KeyType>& assignment() const { return assignment_; }

    //! @brief number of local particles to be sent to lower ranks
    LocalIndex numSendDown() const { return exchanges_[myRank_]; }
    LocalIndex numPresent() const { return exchanges_.count(myRank_); }
    LocalIndex numAssigned() const { return assignment_.totalCount(myRank_); }

private:
    int myRank_;
    int numRanks_;
    unsigned bucketSize_;

    //! @brief global coordinate bounding box
    Box<T> box_;

    SfcAssignment<KeyType> assignment_;
    SendRanges exchanges_;
    mutable ExchangeLog recvLog_;

    //! @brief For locating global domain boundaries in local particle key arrays
    DeviceVector<KeyType> d_boundaryKeys_;
    DeviceVector<LocalIndex> d_boundaryIndices_;

    //! @brief leaf particle counts
    std::vector<unsigned> nodeCounts_;
    DeviceVector<unsigned> d_nodeCounts_;

    //! @brief the fully linked octree
    Octree<KeyType> tree_;
    DeviceVector<KeyType> d_csTree_;

    bool firstCall_{true};
};

} // namespace cstone
