/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
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

#include "cstone/cuda/device_vector.h"
#include "cstone/domain/domaindecomp.hpp"
#include "cstone/domain/domaindecomp_mpi.hpp"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/tree/octree.hpp"
#include "cstone/tree/update_mpi.hpp"
#include "cstone/tree/update_mpi_gpu.cuh"
#include "cstone/sfc/box_mpi.hpp"
#include "cstone/sfc/sfc.hpp"
#include "cstone/sfc/sfc_gpu.h"
#ifdef USE_CUDA
#include "cstone/domain/domaindecomp_gpu.cuh"
#include "cstone/domain/domaindecomp_mpi_gpu.cuh"
#endif

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
template<class KeyType, class T, class Accelerator = CpuTag>
class GlobalAssignment
{
    template<class ValueType>
    using AccVector = std::conditional_t<HaveGpu<Accelerator>{}, DeviceVector<ValueType>, std::vector<ValueType>>;

    constexpr static bool gpu = HaveGpu<Accelerator>{};

public:
    GlobalAssignment(int rank, int nRanks, unsigned bucketSize, const Box<T>& box)
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

        if constexpr (gpu) { reallocate(numRanks_ + 1, 1.0, d_boundaryKeys_, d_boundaryIndices_); }
    }

    /*! @brief Update the global tree
     *
     * @param[in]  o1              buffer description with range of assigned particles
     * @param[in]  reorderFunctor  records the SFC order of the owned input coordinates
     * @param[out] particleKeys    will contain sorted particle SFC keys in the range [bufDesc.start:bufDesc.end]
     * @param[-]   s0              device scratch vector
     * @param[-]   s1              device scratch vector
     * @param[in]  x               x coordinates
     * @param[in]  y               y coordinates
     * @param[in]  z               z coordinates
     * @return                     required buffer size for the next call to @a distribute
     *
     * This function does not modify / communicate any particle data.
     */
    template<class Reorderer, class Vector>
    LocalIndex assign(BufferDescription o1,
                      Reorderer& reorderFunctor,
                      Vector& s0,
                      Vector& s1,
                      KeyType* particleKeys,
                      const T* x,
                      const T* y,
                      const T* z)
    {
        // number of locally assigned particles to consider for global tree building
        LocalIndex numPart = o1.end - o1.start;

        using Op        = std::conditional_t<HaveGpu<Accelerator>{}, MinMaxGpu<T>, MinMax<T>>;
        auto fittingBox = makeGlobalBox<T, Op>(x + o1.start, y + o1.start, z + o1.start, numPart, box_);
        if (firstCall_) { box_ = fittingBox; }
        else { box_ = limitBoxShrinking(fittingBox, box_); }

        // compute SFC particle keys only for particles participating in tree build
        std::span<KeyType> keyView(particleKeys + o1.start, numPart);
        computeSfcKeys<gpu>(x + o1.start, y + o1.start, z + o1.start, sfcKindPointer(keyView.data()), numPart, box_);
        sequence<gpu>(o1.start, numPart, reorderFunctor.getBuf(), growthRate_);
        sortByKey<gpu>(keyView, std::span{reorderFunctor.getMap() + o1.start, keyView.size()}, s0, s1, growthRate_);

        updateOctreeGlobal<gpu, KeyType>(keyView, bucketSize_, tree_, d_csTree_, nodeCounts_, d_nodeCounts_);
        if (firstCall_)
        {
            firstCall_ = false;
            while (
                !updateOctreeGlobal<gpu, KeyType>(keyView, bucketSize_, tree_, d_csTree_, nodeCounts_, d_nodeCounts_))
                ;
        }

        auto newAssignment = makeSfcAssignment(numRanks_, nodeCounts_, tree_.treeLeaves().data());
        limitBoundaryShifts<KeyType>(assignment_, newAssignment, tree_.treeLeaves(), nodeCounts_);
        assignment_ = std::move(newAssignment);

        if constexpr (gpu)
        {
            exchanges_ =
                createSendRangesGpu<KeyType>(assignment_, keyView, rawPtr(d_boundaryKeys_), rawPtr(d_boundaryIndices_));
        }
        else { exchanges_ = createSendRanges<KeyType>(assignment_, keyView); }

        return domain_exchange::exchangeBufferSize(o1, numPresent(), numAssigned());
    }

    /*! @brief Distribute particles to their assigned ranks based on previous assignment
     *
     * @param[in]    o1e                Buffer description with range of assigned particles and total buffer size
     * @param[inout] reorderFunctor     contains the ordering that accesses the range [particleStart:particleEnd]
     *                                  in SFC order
     * @param[-]     s0        scratch space for send buffers
     * @param[-]     s1     scratch space for receive buffers
     * @param[in]    keys               particle SFC keys, sorted in [bufDesc.start:bufDesc.end]
     * @param[inout] x                  particle x-coordinates
     * @param[inout] y                  particle y-coordinates
     * @param[inout] z                  particle z-coordinates
     * @param[inout] properties remaining particle properties, h, m, etc.
     * @return                          index denoting the index range start of particles post-exchange
     *                                  plus a span with a view of the assigned particle keys
     *
     * Note: Instead of reordering the particle buffers right here after the exchange, we only keep track
     * of the reorder map that is required to transform the particle buffers into SFC-order. This allows us
     * to defer the reordering until we have done halo discovery. At that time, we know the final location
     * where to put the assigned particles inside the buffer, such that we can reorder directly to the final
     * location. This saves us from having to move around data inside the buffers for a second time.
     */
    template<class Reorderer, class Vector, class... Arrays>
    auto distribute(BufferDescription o1e,
                    Reorderer& reorderFunctor,
                    Vector& s0,
                    Vector& s1,
                    KeyType* keys,
                    T* x,
                    T* y,
                    T* z,
                    Arrays... properties) const
    {
        recvLog_.clear();

        auto numRecv   = numAssigned() - numPresent();
        auto recvStart = domain_exchange::receiveStart(o1e, numRecv);
        if constexpr (gpu)
        {
            exchangeParticlesGpu(0, recvLog_, exchanges_, myRank_, recvStart, recvStart + numRecv, s0, s1,
                                 reorderFunctor.getMap() + o1e.start, x, y, z, properties...);
        }
        else
        {
            exchangeParticles(0, recvLog_, exchanges_, myRank_, recvStart, recvStart + numRecv,
                              reorderFunctor.getMap() + o1e.start, x, y, z, properties...);
        }

        auto [newStart, newEnd] = domain_exchange::assignedEnvelope(o1e, numAssigned() - numPresent());
        LocalIndex envelopeSize = newEnd - newStart;
        std::span<KeyType> keyView(keys + newStart, envelopeSize);

        computeSfcKeys<gpu>(x + recvStart, y + recvStart, z + recvStart, sfcKindPointer(keys + recvStart), numRecv,
                            box_);
        sequence<gpu>(recvStart, numRecv, reorderFunctor.getBuf(), growthRate_);
        sortByKey<gpu>(keyView, std::span{reorderFunctor.getMap() + newStart, keyView.size()}, s0, s1, growthRate_);

        return std::make_tuple(newStart, keyView.subspan(numSendDown(), numAssigned()));
    }

    //! @brief repeat exchange from last call to assign()
    template<class SVec, class... Arrays>
    auto redoExchange(
        BufferDescription o1e, const LocalIndex* ordering, SVec& /*s1*/, SVec& /*s2*/, Arrays... properties) const
    {
        auto numRecv    = numAssigned() - numPresent();
        auto recvStart  = domain_exchange::receiveStart(o1e, numRecv);
        auto exchangeO2 = shiftSendRanges(exchanges_, myRank_, numRecv);
        exchangeParticles(1, recvLog_, exchangeO2, myRank_, recvStart, recvStart + numRecv, ordering, properties...);
    }

    //! @brief read only visibility of the global octree leaves to the outside
    std::span<const KeyType> treeLeaves() const
    {
        if (gpu) { return {rawPtr(d_csTree_), d_csTree_.size()}; }
        else { return tree_.treeLeaves(); }
    }

    //! @brief read only visibility of the global octree leaf counts to the outside
    std::span<const unsigned> nodeCounts() const
    {
        if (gpu) { return {rawPtr(d_nodeCounts_), d_nodeCounts_.size()}; }
        else { return nodeCounts_; }
    }

    //! @brief the octree, including the internal part
    const Octree<KeyType>& octree() const { return tree_; }
    //! @brief the global coordinate bounding box
    const Box<T>& box() const { return box_; }
    //! @brief return the space filling curve rank assignment of the last call to @a assign()
    const SfcAssignment<KeyType>& assignment() const { return assignment_; }

    LocalIndex o3start(BufferDescription o1) const
    {
        return domain_exchange::assignedEnvelope(o1, numAssigned() - numPresent())[0] + numSendDown();
    }
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

    AccVector<KeyType> d_boundaryKeys_;
    AccVector<LocalIndex> d_boundaryIndices_;

    //! @brief leaf particle counts
    std::vector<unsigned> nodeCounts_;
    AccVector<unsigned> d_nodeCounts_;

    //! @brief the fully linked octree
    Octree<KeyType> tree_;
    AccVector<KeyType> d_csTree_;

    bool firstCall_{true};
    double growthRate_{1.05};
};

} // namespace cstone
