/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
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
 * @brief A domain class to manage distributed particles and their halos.
 *
 * Particles are represented by x,y,z coordinates, interaction radii and
 * a user defined number of additional properties, such as masses or charges.
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/domain/assignment.hpp"
#ifdef USE_CUDA
#include "cstone/domain/assignment_gpu.cuh"
#include "cstone/primitives/gather.cuh"
#endif
#include "cstone/domain/exchange_keys.hpp"
#include "cstone/domain/layout.hpp"
#include "cstone/focus/octree_focus_mpi.hpp"
#include "cstone/halos/halos.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/traversal/collisions.hpp"
#include "cstone/traversal/peers.hpp"
#include "cstone/primitives/accel_switch.hpp"
#include "cstone/sfc/box_mpi.hpp"
#include "cstone/sfc/sfc.hpp"
#include "cstone/sfc/sfc_gpu.h"
#include "cstone/util/reallocate.hpp"
#include "cstone/util/type_list.hpp"

namespace cstone
{

template<class KeyType, class T>
class GlobalAssignmentGpu;

template<class IndexType, class BufferType>
class GpuSfcSorter;

template<class KeyType, class T, class Accelerator = CpuTag>
class Domain
{
    static_assert(std::is_unsigned<KeyType>{}, "SFC key type needs to be an unsigned integer\n");

    //! @brief A vector template that resides on the hardware specified as Accelerator
    template<class ValueType>
    using AccVector = typename AccelSwitchType<Accelerator, std::vector, DeviceVector>::template type<ValueType>;

    template<class BufferType>
    using ReorderFunctor_t =
        typename AccelSwitchType<Accelerator, SfcSorter, GpuSfcSorter>::template type<LocalIndex, BufferType>;

public:
    //! @brief floating point type used for the coordinate bounding box and geometric/mass centers of tree nodes
    using RealType = T;

    /*! @brief construct empty Domain
     *
     * @param rank            executing rank
     * @param nRanks          number of ranks
     * @param bucketSize      build global tree for domain decomposition with max @a bucketSize particles per node
     * @param bucketSizeFocus maximum number of particles per leaf node inside the assigned part of the SFC
     * @param theta           angle parameter to control focus resolution and gravity accuracy
     * @param box             global bounding box, default is non-pbc box
     *                        for each periodic dimension in @a box, the coordinate min/max
     *                        limits will never be changed for the lifetime of the Domain
     *
     */
    Domain(int rank,
           int nRanks,
           unsigned bucketSize,
           unsigned bucketSizeFocus,
           float theta,
           const Box<T>& box = Box<T>{0, 1})
        : myRank_(rank)
        , numRanks_(nRanks)
        , bucketSizeFocus_(bucketSizeFocus)
        , theta_(theta)
        , focusTree_(rank, numRanks_, bucketSizeFocus_)
        , global_(rank, nRanks, bucketSize, box)
    {
        if (bucketSize < bucketSizeFocus_)
        {
            throw std::runtime_error("The bucket size of the global tree must not be smaller than the bucket size"
                                     " of the focused tree\n");
        }
    }

    /*! @brief Domain update sequence for particles with coordinates x,y,z, interaction radius h and their properties
     *
     * @param[out]   particleKeys        SFC particleKeys
     * @param[inout] x                   floating point coordinates
     * @param[inout] y
     * @param[inout] z
     * @param[inout] h                   interaction radii in SPH convention, actual interaction radius
     *                                   is twice the value in h
     * @param[inout] particleProperties  particle properties to distribute along with the coordinates
     *                                   e.g. mass or charge
     *
     * ============================================================================================================
     * Preconditions:
     * ============================================================================================================
     *
     *   - Array sizes of x,y,z,h and particleProperties are identical
     *     AND equal to the internally stored value of localNParticles_ from the previous call, except
     *     on the first call. This is checked.
     *
     *     This means that none of the argument arrays can be resized between calls of this function.
     *     Or in other words, particles cannot be created or destroyed.
     *     (If this should ever be required though, it can be easily enabled by allowing the assigned
     *     index range from startIndex() to endIndex() to be modified from the outside.)
     *
     *   - The particle order is irrelevant
     *
     *   - Content of particleKeys is irrelevant as it will be resized to fit x,y,z,h and particleProperties
     *
     * ============================================================================================================
     * Postconditions:
     * ============================================================================================================
     *
     *   Array sizes:
     *   ------------
     *   - All arrays, x,y,z,h, particleKeys and particleProperties are resized with space for the newly assigned
     *     particles AND their halos.
     *
     *   Content of x,y,z and h
     *   ----------------------------
     *   - x,y,z,h at indices from startIndex() to endIndex() contain assigned particles that the executing rank owns,
     *     all other elements are _halos_ of the assigned particles, i.e. the halos for x,y,z,h and particleKeys are
     *     already in place post-call.
     *
     *   Content of particleProperties
     *   ----------------------------
     *   - particleProperties arrays contain the updated properties at indices from startIndex() to endIndex(),
     *     i.e. index i refers to a property of the particle with coordinates (x[i], y[i], z[i]).
     *     Content of elements outside this range is _undefined_, but can be filled with the corresponding halo data
     *     by a subsequent call to exchangeHalos(particleProperty), such that also for i outside
     *     [startIndex():endIndex()], particleProperty[i] is a property of the halo particle with
     *     coordinates (x[i], y[i], z[i]).
     *
     *   Content of particleKeys
     *   ----------------
     *   - The particleKeys output is sorted and contains the SFC particleKeys of assigned _and_ halo particles,
     *     i.e. all arrays will be output in SFC order.
     *
     *   Internal state of the domain
     *   ----------------------------
     *   The following members are modified by calling this function:
     *   - Update of the global octree, for use as starting guess in the next call
     *   - Update of the assigned range startIndex() and endIndex()
     *   - Update of the total local particle count, i.e. assigned + halo particles
     *   - Update of the halo exchange patterns, for subsequent use in exchangeHalos
     *   - Update of the global coordinate bounding box
     *
     * ============================================================================================================
     * Update sequence:
     * ============================================================================================================
     *      1. compute global coordinate bounding box
     *      2. compute global octree
     *      3. compute max_h per octree node
     *      4. assign octree to ranks
     *      5. discover halos
     *      6. compute particle layout, i.e. count number of halos and assigned particles
     *         and compute halo send and receive index ranges
     *      7. resize x,y,z,h,particleKeys and properties to new number of assigned + halo particles
     *      8. exchange coordinates, h, and properties of assigned particles
     *      9. SFC sort exchanged assigned particles
     *     10. exchange halo particles
     */
    template<class KeyVec, class VectorX, class VectorH, class... Vectors1, class... Vectors2>
    void sync(KeyVec& particleKeys,
              VectorX& x,
              VectorX& y,
              VectorX& z,
              VectorH& h,
              std::tuple<Vectors1&...> particleProperties,
              std::tuple<Vectors2&...> scratchBuffers)
    {
        staticChecks<KeyVec, VectorX, VectorH, Vectors1...>(scratchBuffers);
        auto& sfcOrder = std::get<sizeof...(Vectors2) - 1>(scratchBuffers);
        ReorderFunctor_t<std::decay_t<decltype(sfcOrder)>> sorter(sfcOrder);

        auto scratch = util::discardLastElement(scratchBuffers);

        auto [exchangeStart, keyView] =
            distribute(sorter, particleKeys, x, y, z, std::tuple_cat(std::tie(h), particleProperties), scratch);
        // h is already reordered here for use in halo discovery
        gatherArrays(sorter.gatherFunc(), sorter.getMap() + global_.numSendDown(), global_.numAssigned(), exchangeStart,
                     0, std::tie(h), util::reverse(scratch));

        float invThetaEff      = invThetaMinMac(theta_);
        std::vector<int> peers = findPeersMac(myRank_, global_.assignment(), global_.octree(), box(), invThetaEff);

        if (firstCall_)
        {
            focusTree_.converge(box(), keyView, peers, global_.assignment(), global_.treeLeaves(), global_.nodeCounts(),
                                invThetaEff, std::get<0>(scratch));
        }
        focusTree_.updateMinMac(global_.assignment(), invThetaEff);
        focusTree_.updateTree(peers, global_.assignment(), box());
        focusTree_.updateCounts(keyView, global_.treeLeaves(), global_.nodeCounts(), std::get<0>(scratch));

        auto octreeView            = focusTree_.octreeViewAcc();
        const KeyType* focusLeaves = focusTree_.treeLeavesAcc().data();

        reallocateDestructive(layout_, octreeView.numLeafNodes + 1, allocGrowthRate_);
        reallocateDestructive(layoutAcc_, octreeView.numLeafNodes + 1, allocGrowthRate_);
        halos_.discover(octreeView.prefixes, octreeView.childOffsets, octreeView.internalToLeaf, focusLeaves,
                        focusTree_.leafCountsAcc(), focusTree_.assignment(), {rawPtr(layoutAcc_), layoutAcc_.size()},
                        box(), rawPtr(h), haloSearchExt_, std::get<0>(scratch));
        halos_.computeLayout(focusTree_.treeLeaves(), focusTree_.leafCounts(), focusTree_.assignment(), peers, layout_);

        updateLayout(sorter, exchangeStart, keyView, particleKeys, std::tie(h),
                     std::tuple_cat(std::tie(x, y, z), particleProperties), scratch);
        setupHalos(particleKeys, x, y, z, h, scratch);
        firstCall_ = false;
    }

    template<class KeyVec, class VectorX, class VectorH, class VectorM, class... Vectors1, class... Vectors2>
    void syncGrav(KeyVec& particleKeys,
                  VectorX& x,
                  VectorX& y,
                  VectorX& z,
                  VectorH& h,
                  VectorM& m,
                  std::tuple<Vectors1&...> particleProperties,
                  std::tuple<Vectors2&...> scratchBuffers)
    {
        staticChecks<KeyVec, VectorX, VectorH, VectorM, Vectors1...>(scratchBuffers);
        auto& sfcOrder = std::get<sizeof...(Vectors2) - 1>(scratchBuffers);
        ReorderFunctor_t<std::decay_t<decltype(sfcOrder)>> sorter(sfcOrder);

        auto scratch = util::discardLastElement(scratchBuffers);

        auto [exchangeStart, keyView] =
            distribute(sorter, particleKeys, x, y, z, std::tuple_cat(std::tie(h, m), particleProperties), scratch);
        gatherArrays(sorter.gatherFunc(), sorter.getMap() + global_.numSendDown(), global_.numAssigned(), exchangeStart,
                     0, std::tie(x, y, z, h, m), util::reverse(scratch));

        float invThetaEff      = invThetaVecMac(theta_);
        std::vector<int> peers = findPeersMac(myRank_, global_.assignment(), global_.octree(), box(), invThetaEff);

        if (firstCall_)
        {
            // first rough convergence to avoid computing expansion centers of large nodes with a lot of particles
            focusTree_.converge(box(), keyView, peers, global_.assignment(), global_.treeLeaves(), global_.nodeCounts(),
                                1.0, std::get<0>(scratch));
            focusTree_.updateMinMac(global_.assignment(), 1.0);
            int converged = 0, reps = 0;
            while (converged != numRanks_ || reps < 2)
            {
                converged = focusTree_.updateTree(peers, global_.assignment(), box());
                focusTree_.updateCounts(keyView, global_.treeLeaves(), global_.nodeCounts(), std::get<0>(scratch));
                focusTree_.updateCenters(rawPtr(x), rawPtr(y), rawPtr(z), rawPtr(m), global_.octree(),
                                         std::get<0>(scratch), std::get<1>(scratch));
                focusTree_.updateMacs(global_.assignment(), 1.0 / theta_);
                MPI_Allreduce(MPI_IN_PLACE, &converged, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                reps++;
            }
        }

        int fail = 0;
        do
        {
            focusTree_.updateMacs(global_.assignment(), centerDriftTol_ / theta_);
            focusTree_.updateTree(peers, global_.assignment(), box());
            focusTree_.updateCounts(keyView, global_.treeLeaves(), global_.nodeCounts(), std::get<0>(scratch));
            focusTree_.updateCenters(rawPtr(x), rawPtr(y), rawPtr(z), rawPtr(m), global_.octree(), std::get<0>(scratch),
                                     std::get<1>(scratch));
            focusTree_.updateMacs(global_.assignment(), 1.0 / theta_);

            auto octreeView            = focusTree_.octreeViewAcc();
            const KeyType* focusLeaves = focusTree_.treeLeavesAcc().data();

            reallocateDestructive(layout_, octreeView.numLeafNodes + 1, allocGrowthRate_);
            reallocateDestructive(layoutAcc_, octreeView.numLeafNodes + 1, allocGrowthRate_);
            halos_.discover(octreeView.prefixes, octreeView.childOffsets, octreeView.internalToLeaf, focusLeaves,
                            focusTree_.leafCountsAcc(), focusTree_.assignment(),
                            {rawPtr(layoutAcc_), layoutAcc_.size()}, box(), rawPtr(h), haloSearchExt_,
                            std::get<0>(scratch));
            focusTree_.addMacs(halos_.haloFlags());
            fail = halos_.computeLayout(focusTree_.treeLeaves(), focusTree_.leafCounts(), focusTree_.assignment(),
                                        peers, layout_);
            MPI_Allreduce(MPI_IN_PLACE, &fail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

            if (fail)
            {
                centerDriftTol_ += 0.05;
                if (myRank_ == 0) { std::cout << "Increased centerDriftTol to " << centerDriftTol_ << std::endl; }
            }
        } while (fail);

        // diagnostics(keyView.size(), peers);

        updateLayout(sorter, exchangeStart, keyView, particleKeys, std::tie(x, y, z, h, m), particleProperties,
                     scratch);
        setupHalos(particleKeys, x, y, z, h, scratch);
        firstCall_ = false;
    }

    /*! @brief reapply exchange synchronization pattern from previous call to sync(Grav)() to additional particle fields
     *
     * @param[inout] arrays          the arrays to reapply sync to, length prevBufDesc_.size
     * @param[-]     sendBuffer
     * @param[-]     receiveBuffer
     * @param[in]    ordering        the post-particle-exchange SFC ordering
     */
    template<class... Vectors, class SendBuffer, class ReceiveBuffer, class OVec>
    void reapplySync(std::tuple<Vectors&...> arrays,
                     SendBuffer& sendBuffer,
                     ReceiveBuffer& receiveBuffer,
                     OVec& ordering) const
    {
        static_assert((... && !IsDeviceVector<Vectors>{}), "reapplySync only support for arrays on CPUs");
        std::apply([this](auto&... arrays) { this->template checkSizesEqual(this->prevBufDesc_.size, arrays...); },
                   arrays);

        LocalIndex exSize =
            domain_exchange::exchangeBufferSize(prevBufDesc_, global_.numPresent(), global_.numAssigned());
        lowMemReallocate(exSize, allocGrowthRate_, arrays, {});

        BufferDescription exDesc{prevBufDesc_.start, prevBufDesc_.end, exSize};
        auto envelope    = domain_exchange::assignedEnvelope(exDesc, global_.numPresent(), global_.numAssigned());
        LocalIndex shift = prevBufDesc_.start - envelope[0];

        // the intermediate, reconstructed ordering needed for the MPI particle exchange
        std::vector<LocalIndex> prevOrd(prevBufDesc_.end - prevBufDesc_.start);
        // the post-exchange ordering that was obtained by sorting after receiving the particles from domain exchange
        std::vector<LocalIndex> orderingCpu;

        auto* ord = (LocalIndex*)rawPtr(ordering);
        if constexpr (HaveGpu<Accelerator>{})
        {
            static_assert(IsDeviceVector<OVec>{}, "Need ordering on GPU for GPU-accelerated domain");
            orderingCpu.resize(envelope[1] - envelope[0]);
            memcpyD2H((LocalIndex*)rawPtr(ordering), orderingCpu.size(), orderingCpu.data());
            ord = orderingCpu.data();
        }

        std::transform(ord, ord + global_.numSendDown(), prevOrd.data(), [shift](auto i) { return i - shift; });
        std::transform(ord + global_.numSendDown() + global_.numAssigned(), ord + envelope[1] - envelope[0],
                       prevOrd.data() + global_.numSendDown() + global_.numPresent(),
                       [shift](auto i) { return i - shift; });

        std::apply([exDesc, o = prevOrd.data(), &sendBuffer, &receiveBuffer, this](auto&... a)
                   { global_.redoExchange(exDesc, o, sendBuffer, receiveBuffer, rawPtr(a)...); },
                   arrays);

        lowMemReallocate(bufDesc_.size, allocGrowthRate_, arrays, std::tie(sendBuffer, receiveBuffer));
        gatherArrays(gatherCpu, ord + global_.numSendDown(), global_.numAssigned(), envelope[0], bufDesc_.start, arrays,
                     std::tie(sendBuffer, receiveBuffer));
    }

    //! @brief repeat the halo exchange pattern from the previous sync operation for a different set of arrays
    template<class... Vectors, class SendBuffer, class ReceiveBuffer>
    void exchangeHalos(std::tuple<Vectors&...> arrays, SendBuffer& sendBuffer, ReceiveBuffer& receiveBuffer) const
    {
        std::apply([this](auto&... arrays) { this->template checkSizesEqual(this->bufDesc_.size, arrays...); }, arrays);
        this->halos_.exchangeHalos(arrays, sendBuffer, receiveBuffer);
    }

    //! @brief return the index of the first particle that's part of the local assignment
    [[nodiscard]] LocalIndex startIndex() const { return bufDesc_.start; }
    //! @brief return one past the index of the last particle that's part of the local assignment
    [[nodiscard]] LocalIndex endIndex() const { return bufDesc_.end; }
    //! @brief set the index of the lsat particle (used to increase the number of particles)
    void setEndIndex(const size_t i) { bufDesc_.end = i; }
    //! @brief return number of locally assigned particles
    [[nodiscard]] LocalIndex nParticles() const { return endIndex() - startIndex(); }
    //! @brief return number of locally assigned particles plus number of halos
    [[nodiscard]] LocalIndex nParticlesWithHalos() const { return bufDesc_.size; }
    //! @brief read only visibility of the global octree leaves to the outside
    const Octree<KeyType>& globalTree() const { return global_.octree(); }
    //! @brief read only visibility of the focused octree
    const FocusedOctree<KeyType, T, Accelerator>& focusTree() const { return focusTree_; }
    //! @brief the index of the first locally assigned cell in focusTree()
    TreeNodeIndex startCell() const { return focusTree_.assignment()[myRank_].start(); }
    //! @brief the index of the last locally assigned cell in focusTree()
    TreeNodeIndex endCell() const { return focusTree_.assignment()[myRank_].end(); }
    //! @brief particle offsets of each focus tree leaf cell
    gsl::span<const LocalIndex> layout() const { return {rawPtr(layoutAcc_), layoutAcc_.size()}; }
    //! @brief return the coordinate bounding box from the previous sync call
    const Box<T>& box() const { return global_.box(); }

    void setTreeConv(bool flag) { convergeTrees = flag; }
    void setHaloFactor(float factor) { haloSearchExt_ = factor; }
    void setGrowthAllocRate(float factor) { allocGrowthRate_ = factor; }

    //! @brief update expansion (c.o.m) centers of the focus tree
    template<class VectorX, class VectorM, class VectorS1, class VectorS2>
    void updateExpansionCenters(VectorX& x, VectorX& y, VectorX& z, VectorM& m, VectorS1& s1, VectorS2& s2)
    {
        auto si = startIndex();
        focusTree_.updateCenters(rawPtr(x) + si, rawPtr(y) + si, rawPtr(z) + si, rawPtr(m) + si, global_.octree(), s1,
                                 s2);
        focusTree_.setMacRadius(1.0 / theta_);
    };

    OctreeNsView<T, KeyType> octreeProperties() const
    {
        auto ft = focusTree_.octreeViewAcc();
        return {ft.numLeafNodes,
                ft.prefixes,
                ft.childOffsets,
                ft.internalToLeaf,
                ft.levelRange,
                focusTree_.treeLeavesAcc().data(),
                rawPtr(layoutAcc_),
                focusTree_.geoCentersAcc().data(),
                focusTree_.geoSizesAcc().data()};
    }

private:
    //! @brief bounds initialization on first call, use all particles
    void initBounds(std::size_t bufferSize)
    {
        if (firstCall_)
        {
            bufDesc_     = {0, LocalIndex(bufferSize), LocalIndex(bufferSize)};
            prevBufDesc_ = bufDesc_;
            layout_      = {0, LocalIndex(bufferSize)};
        }
    }

    //! @brief make sure all array sizes are equal to @p value
    template<class... Arrays>
    static void checkSizesEqual(std::size_t value, const Arrays&... arrays)
    {
        std::array<std::size_t, sizeof...(Arrays)> sizes{arrays.size()...};
        bool allEqual = size_t(std::count(begin(sizes), end(sizes), value)) == sizes.size();
        if (!allEqual) { throw std::runtime_error("Domain sync: input array sizes are inconsistent\n"); }
    }

    /*! @brief check type requirements on scratch buffers
     *
     * @tparam KeyVec           type of vector used to store SFC keys
     * @tparam ConservedVectors types of conserved particle field vectors (x,y,z,...)
     * @param  scratchBuffers   a tuple of references to vectors for scratch usage
     *
     * At least 3 scratch buffers are needed. 2 for send/receive and the last one is used to store the SFC ordering.
     * An additional requirement is that for each value type appearing in the list of conserved
     * vectors, either a scratch buffer with a matching value_type (more efficient due to swaps) or with a value_type
     * of equal or bigger size (less efficient due an additional copy) is needed.
     */
    template<class KeyVec, class... ConservedVectors, class ScratchBuffers>
    void staticChecks(ScratchBuffers& scratchBuffers)
    {
        static_assert(std::is_same_v<typename KeyVec::value_type, KeyType>);
        static_assert(std::tuple_size_v<ScratchBuffers> >= 3);

        auto tup               = util::discardLastElement(scratchBuffers);
        constexpr auto matches = std::make_tuple(util::FindIndex<ConservedVectors&, std::decay_t<decltype(tup)>>{}...);
        constexpr auto smaller =
            std::make_tuple(util::FindIndex<ConservedVectors&, std::decay_t<decltype(tup)>, SmallerElementSize>{}...);

        auto valueTypeCheck = [](auto index)
        {
            constexpr int numScratchBuffers = std::tuple_size_v<std::decay_t<decltype(tup)>>;
            static_assert(get<0>(index) < numScratchBuffers || get<1>(index) < numScratchBuffers,
                          "one of the conserved fields has a value_type bigger than the value_types of available "
                          "scratch buffers");
        };
        util::for_each_tuple(valueTypeCheck, util::zipTuples(matches, smaller));
    }

    template<class Sorter, class KeyVec, class VectorX, class... Vectors1, class... Vectors2>
    auto distribute(Sorter& sorter,
                    KeyVec& keys,
                    VectorX& x,
                    VectorX& y,
                    VectorX& z,
                    std::tuple<Vectors1&...> particleProperties,
                    std::tuple<Vectors2&...> scratchBuffers)
    {
        initBounds(x.size());
        auto distributedArrays = std::tuple_cat(std::tie(keys, x, y, z), particleProperties);
        std::apply([size = x.size()](auto&... arrays) { checkSizesEqual(size, arrays...); }, distributedArrays);

        // Global tree build and assignment
        auto exchangeSize = global_.assign(bufDesc_, sorter, std::get<0>(scratchBuffers), std::get<1>(scratchBuffers),
                                           rawPtr(keys), rawPtr(x), rawPtr(y), rawPtr(z));
        lowMemReallocate(exchangeSize, allocGrowthRate_, distributedArrays, scratchBuffers);

        // Must zero new memory to exclude possibility of special value (removeKey) in uninitialized memory
        fill<IsDeviceVector<KeyVec>{}>(rawPtr(keys) + bufDesc_.size, rawPtr(keys) + exchangeSize, KeyType(0));

        return std::apply(
            [exchangeSize, &sorter, &scratchBuffers, this](auto&... arrays)
            {
                return global_.distribute({bufDesc_.start, bufDesc_.end, exchangeSize}, sorter,
                                          std::get<0>(scratchBuffers), std::get<1>(scratchBuffers), rawPtr(arrays)...);
            },
            distributedArrays);
    }

    template<class KeyVec, class VectorX, class VectorH, class... Vs>
    void setupHalos(KeyVec& keys, VectorX& x, VectorX& y, VectorX& z, VectorH& h, std::tuple<Vs&...> scratch)
    {
        exchangeHalos(std::tie(x, y, z, h), std::get<0>(scratch), std::get<1>(scratch));

        // compute SFC keys of received halo particles
        if constexpr (IsDeviceVector<KeyVec>{})
        {
            computeSfcKeysGpu(rawPtr(x), rawPtr(y), rawPtr(z), sfcKindPointer(rawPtr(keys)), bufDesc_.start, box());
            computeSfcKeysGpu(rawPtr(x) + bufDesc_.end, rawPtr(y) + bufDesc_.end, rawPtr(z) + bufDesc_.end,
                              sfcKindPointer(rawPtr(keys)) + bufDesc_.end, x.size() - bufDesc_.end, box());
        }
        else
        {
            computeSfcKeys(rawPtr(x), rawPtr(y), rawPtr(z), sfcKindPointer(rawPtr(keys)), bufDesc_.start, box());
            computeSfcKeys(rawPtr(x) + bufDesc_.end, rawPtr(y) + bufDesc_.end, rawPtr(z) + bufDesc_.end,
                           sfcKindPointer(rawPtr(keys)) + bufDesc_.end, x.size() - bufDesc_.end, box());
        }
    }

    template<class Sorter, class KeyVec, class... Arrays1, class... Arrays2, class... Arrays3>
    void updateLayout(Sorter& sorter,
                      LocalIndex exchangeStart,
                      gsl::span<const KeyType> keyView,
                      KeyVec& keys,
                      std::tuple<Arrays1&...> orderedBuffers,
                      std::tuple<Arrays2&...> unorderedBuffers,
                      std::tuple<Arrays3&...> scratchBuffers)
    {
        auto myRange = focusTree_.assignment()[myRank_];
        BufferDescription newBufDesc{layout_[myRange.start()], layout_[myRange.end()], layout_.back()};

        lowMemReallocate(newBufDesc.size, allocGrowthRate_, std::tuple_cat(orderedBuffers, unorderedBuffers),
                         scratchBuffers);

        // re-locate particle SFC keys
        if constexpr (IsDeviceVector<KeyVec>{})
        {
            memcpyH2D(layout_.data(), layout_.size(), rawPtr(layoutAcc_));
            auto& swapSpace = std::get<0>(scratchBuffers);
            size_t origSize = reallocateBytes(swapSpace, keyView.size() * sizeof(KeyType), allocGrowthRate_);

            auto* swapPtr = reinterpret_cast<KeyType*>(rawPtr(swapSpace));
            memcpyD2D(keyView.data(), keyView.size(), swapPtr);
            reallocate(keys, newBufDesc.size, allocGrowthRate_);
            fill<true>(rawPtr(keys) + bufDesc_.size, rawPtr(keys) + newBufDesc.size, KeyType(0));
            memcpyD2D(swapPtr, keyView.size(), rawPtr(keys) + newBufDesc.start);

            reallocate(swapSpace, origSize, 1.0);
        }
        else
        {
            omp_copy(layout_.begin(), layout_.end(), layoutAcc_.begin());
            reallocate(swapKeys_, newBufDesc.size, allocGrowthRate_);
            fill<false>(rawPtr(swapKeys_) + bufDesc_.size, rawPtr(swapKeys_) + newBufDesc.size, KeyType(0));
            omp_copy(keyView.begin(), keyView.end(), swapKeys_.begin() + newBufDesc.start);
            swap(keys, swapKeys_);
        }

        // relocate ordered buffer contents from offset 0 to offset newBufDesc.start
        auto relocate =
            [size = keyView.size(), dest = newBufDesc.start, scratch = util::reverse(scratchBuffers)](auto& array)
        {
            static_assert(util::FindIndex<decltype(array), std::tuple<Arrays3&...>>{} < sizeof...(Arrays3),
                          "No suitable scratch buffer available");
            auto& swapSpace = util::pickType<decltype(array)>(scratch);
            if constexpr (IsDeviceVector<std::decay_t<decltype(array)>>{})
            {
                memcpyD2D(rawPtr(array), size, rawPtr(swapSpace) + dest);
            }
            else { omp_copy(array.begin(), array.begin() + size, swapSpace.begin() + dest); }
            swap(array, swapSpace);
        };
        util::for_each_tuple(relocate, orderedBuffers);

        // reorder the unordered buffers
        gatherArrays(sorter.gatherFunc(), sorter.getMap() + global_.numSendDown(), global_.numAssigned(), exchangeStart,
                     newBufDesc.start, unorderedBuffers, util::reverse(scratchBuffers));

        // newBufDesc is now the valid buffer description
        prevBufDesc_ = bufDesc_;
        bufDesc_     = newBufDesc;
    }

    void diagnostics(size_t assignedSize, gsl::span<int> peers)
    {
        auto focusAssignment = focusTree_.assignment();
        auto focusTree       = focusTree_.treeLeaves();
        auto globalTree      = global_.treeLeaves();

        TreeNodeIndex numFocusPeers    = 0;
        TreeNodeIndex numFocusTruePeer = 0;
        for (int i = 0; i < numRanks_; ++i)
        {
            if (i != myRank_)
            {
                numFocusPeers += focusAssignment[i].count();
                for (TreeNodeIndex fi = focusAssignment[i].start(); fi < focusAssignment[i].end(); ++fi)
                {
                    KeyType fnstart  = focusTree[fi];
                    KeyType fnend    = focusTree[fi + 1];
                    TreeNodeIndex gi = findNodeAbove(globalTree, fnstart);
                    if (!(gi < nNodes(globalTree) && globalTree[gi] == fnstart && globalTree[gi + 1] <= fnend))
                    {
                        numFocusTruePeer++;
                    }
                }
            }
        }

        int numFlags = std::count(halos_.haloFlags().cbegin(), halos_.haloFlags().cend(), 1);
        for (int i = 0; i < numRanks_; ++i)
        {
            if (i == myRank_)
            {
                std::cout << "rank " << i << " " << assignedSize << " " << layout_.back()
                          << " focus h/true/peers/loc/tot: " << numFlags << "/" << numFocusTruePeer << "/"
                          << numFocusPeers << "/" << focusAssignment[myRank_].count() << "/"
                          << halos_.haloFlags().size() << " peers: [" << peers.size() << "] ";
                if (numRanks_ <= 32)
                {
                    for (auto r : peers)
                    {
                        std::cout << r << " ";
                    }
                }
                std::cout << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    int myRank_;
    int numRanks_;
    unsigned bucketSizeFocus_;

    //! @brief MAC parameter for focus resolution and gravity treewalk
    float theta_;

    bool convergeTrees{false};
    //! @brief Extra search factor for halo discovery, allowing multiple time integration steps between sync() calls
    float haloSearchExt_{1.0};
    //! @brief factor to tighten theta to avoid failed macs by remote cells due to centers having moved closer
    float centerDriftTol_{1.05};
    //! @brief buffer growth rate when reallocating
    float allocGrowthRate_{1.05};

    /*! @brief description of particle buffers, storing start and end indices of assigned particles and total size
     *
     *  First element: array index of first local particle belonging to the assignment
     *  i.e. the index of the first particle that belongs to this rank and is not a halo
     *  Second element: index (upper bound) of last particle that belongs to the assignment
     */
    BufferDescription prevBufDesc_{0, 0, 0}, bufDesc_{0, 0, 0};

    /*! @brief locally focused, fully traversable octree, used for halo discovery and exchange
     *
     * -Uses bucketSizeFocus_ as the maximum particle count per leaf within the focused SFC area.
     * -Outside the focus area, each leaf node with a particle count larger than bucketSizeFocus_
     *  fulfills a MAC with theta as the opening parameter
     * -Also contains particle counts.
     */
    FocusedOctree<KeyType, T, Accelerator> focusTree_;

    //! @brief particle offsets of each leaf node in focusedTree_, length = focusedTree_.treeLeaves().size()
    AccVector<LocalIndex> layoutAcc_;
    std::vector<LocalIndex> layout_;

    using Distributor_t =
        typename AccelSwitchType<Accelerator, GlobalAssignment, GlobalAssignmentGpu>::template type<KeyType, T>;
    Distributor_t global_;

    Halos<KeyType, Accelerator> halos_{myRank_};

    bool firstCall_{true};

    std::vector<KeyType> swapKeys_;
};

} // namespace cstone
