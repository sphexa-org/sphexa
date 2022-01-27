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
 * @brief Implementation of global particle assignment and distribution
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/domain/domaindecomp_mpi.hpp"
#include "cstone/domain/layout.hpp"
#include "cstone/tree/octree_internal.hpp"
#include "cstone/tree/octree_mpi.hpp"

#include "cstone/sfc/box_mpi.hpp"
#include "cstone/sfc/sfc.hpp"

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
class GlobalAssignment
{
public:
    GlobalAssignment(int rank, int nRanks, unsigned bucketSize, const Box<T>& box = Box<T>{0, 1})
        : myRank_(rank)
        , numRanks_(nRanks)
        , bucketSize_(bucketSize)
        , box_(box)
    {
        std::vector<KeyType> init{0, nodeRange<KeyType>(0)};
        tree_.update(init.data(), nNodes(init));
        nodeCounts_ = std::vector<unsigned>{bucketSize_ + 1};
    }

    /*! @brief Update the global tree
     *
     * @param[in]  bufDesc         Buffer description with range of assigned particles
     * @param[in]  reorderFunctor  records the SFC order of the owned input coordinates
     * @param[out] particleKeys    will contain sorted particle SFC keys in the range [particleStart:particleEnd]
     * @param[in]  x               x coordinates
     * @param[in]  y               y coordinates
     * @param[in]  z               z coordinates
     * @return                     number of assigned particles
     *
     * This function does not modify / communicate any particle data.
     */
    template<class Reorderer>
    LocalIndex assign(BufferDescription bufDesc,
                      Reorderer& reorderFunctor,
                      KeyType* particleKeys,
                      const T* x,
                      const T* y,
                      const T* z)
    {
        box_ = makeGlobalBox(x + bufDesc.start, x + bufDesc.end, y + bufDesc.start, z + bufDesc.start, box_);

        // number of locally assigned particles to consider for global tree building
        LocalIndex numParticles = bufDesc.end - bufDesc.start;

        gsl::span<KeyType> keyView(particleKeys + bufDesc.start, numParticles);

        // compute SFC particle keys only for particles participating in tree build
        computeSfcKeys(x + bufDesc.start, y + bufDesc.start, z + bufDesc.start,
                       sfcKindPointer(keyView.data()), numParticles, box_);

        // sort keys and keep track of ordering for later use
        reorderFunctor.setMapFromCodes(keyView.begin(), keyView.end());

        updateOctreeGlobal(keyView.begin(), keyView.end(), bucketSize_, tree_, nodeCounts_, numRanks_);

        if (firstCall_)
        {
            firstCall_ = false;
            while(!updateOctreeGlobal(keyView.begin(), keyView.end(), bucketSize_, tree_, nodeCounts_, numRanks_));
        }

        assignment_ = singleRangeSfcSplit(nodeCounts_, numRanks_);
        return assignment_.totalCount(myRank_);
    }

    /*! @brief Distribute particles to their assigned ranks based on previous assignment
     *
     * @param[in]    bufDesc            Buffer description with range of assigned particles and total buffer size
     * @param[inout] reorderFunctor     contains the ordering that accesses the range [particleStart:particleEnd]
     *                                  in SFC order
     * @param[out]   sfcOrder           If using the CPU reorderer, this is a duplicate copy. Otherwise provides
     *                                  the host space to download the ordering from the device.
     * @param[in]    keys               Sorted particle keys in [particleStart:particleEnd]
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
    template<class Reorderer, class... Arrays>
    auto distribute(BufferDescription bufDesc,
                    Reorderer& reorderFunctor,
                    KeyType* keys,
                    T* x,
                    T* y,
                    T* z,
                    Arrays... particleProperties) const
    {
        LocalIndex numParticles          = bufDesc.end - bufDesc.start;
        LocalIndex newNParticlesAssigned = assignment_.totalCount(myRank_);

        reallocate(numParticles, sfcOrder_);
        reorderFunctor.getReorderMap(sfcOrder_.data(), 0, numParticles);

        gsl::span<KeyType> keyView(keys + bufDesc.start, numParticles);
        SendList domainExchangeSends = createSendList<KeyType>(assignment_, tree_.treeLeaves(), keyView);

        // Assigned particles are now inside the [particleStart:particleEnd] range, but not exclusively.
        // Leftover particles from the previous step can also be contained in the range.
        auto [newStart, newEnd] =
            exchangeParticles(domainExchangeSends, myRank_, bufDesc.start, bufDesc.end, bufDesc.size,
                              newNParticlesAssigned, sfcOrder_.data(), x, y, z, particleProperties...);

        LocalIndex envelopeSize = newEnd - newStart;
        keyView                 = gsl::span<KeyType>(keys + newStart, envelopeSize);

        computeSfcKeys(x + newStart, y + newStart, z + newStart, sfcKindPointer(keyView.begin()), envelopeSize, box_);
        // sort keys and keep track of the ordering
        reorderFunctor.setMapFromCodes(keyView.begin(), keyView.end());

        // thanks to the sorting, we now know the exact range of the assigned particles:
        // [newStart + offset, newStart + offset + newNParticlesAssigned]
        LocalIndex offset = findNodeAbove<KeyType>(keyView, tree_.treeLeaves()[assignment_.firstNodeIdx(myRank_)]);
        // restrict the reordering to take only the assigned particles into account and ignore the others
        reorderFunctor.restrictRange(offset, newNParticlesAssigned);

        return std::make_tuple(newStart, keyView.subspan(offset, newNParticlesAssigned));
    }

    std::vector<int> findPeers(float theta)
    {
        return findPeersMac(myRank_, assignment_, tree_, box_, theta);
    }

    //! @brief read only visibility of the global octree leaves to the outside
    gsl::span<const KeyType> tree() const { return tree_.treeLeaves(); }

    //! @brief read only visibility of the global octree leaf counts to the outside
    gsl::span<const unsigned> nodeCounts() const { return nodeCounts_; }

    //! @brief the global coordinate bounding box
    const Box<T>& box() const { return box_; }

    //! @brief return the space filling curve rank assignment
    const SpaceCurveAssignment& assignment() const { return assignment_; }

private:
    int myRank_;
    int numRanks_;
    unsigned bucketSize_;

    //! @brief global coordinate bounding box
    Box<T> box_;

    SpaceCurveAssignment assignment_;

    //! @brief storage for downloading the sfc ordering from the GPU
    mutable std::vector<LocalIndex> sfcOrder_;

    //! @brief leaf particle counts
    std::vector<unsigned> nodeCounts_;
    //! @brief the fully linked octree
    Octree<KeyType> tree_;

    bool firstCall_{true};
};

} // namespace cstone
