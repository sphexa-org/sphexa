/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
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
#include "cstone/domain/layout.hpp"
#include "cstone/focus/octree_focus_mpi.hpp"
#include "cstone/halos/halos.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/traversal/peers.hpp"
#include "cstone/sfc/box_mpi.hpp"
#include "cstone/sfc/sfc.hpp"
#include "cstone/sfc/sfc_gpu.h"
#include "cstone/util/reallocate.hpp"
#include "cstone/util/type_list.hpp"

namespace cstone
{

template<class KeyType, class T, class Accelerator = CpuTag>
class Domain
{
    static_assert(std::is_unsigned<KeyType>{}, "SFC key type needs to be an unsigned integer\n");

    //! @brief A vector template that resides on the hardware specified as Accelerator
    template<class ValueType>
    using AccVector = std::conditional_t<HaveGpu<Accelerator>{}, DeviceVector<ValueType>, std::vector<ValueType>>;

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
        SfcSorter sorter(sfcOrder);

        auto scratch = util::discardLastElement(scratchBuffers);

        auto [exchangeStart, keyView] =
            distribute(sorter, particleKeys, x, y, z, std::tuple_cat(std::tie(h), particleProperties), scratch);
        // x,y,z,h is already reordered here for use in halo discovery
        gatherArrays({sorter.getMap() + global_.postExchangeStart(bufDesc_), global_.numAssigned()}, 0,
                     std::tie(x, y, z, h), util::reverse(scratch));

        std::vector<int> peers = findPeersMac(myRank_, global_.assignment(), global_.octreeHost(), box(), 1.0 / theta_);
        float invThetaEff      = invThetaMinMac(theta_);

        if (firstCall_)
        {
            focusTree_.converge(box(), keyView, peers, global_.assignment(), global_.treeLeaves(), global_.nodeCounts(),
                                invThetaEff, std::get<0>(scratch));
        }
        focusTree_.updateMinMac(global_.assignment(), invThetaEff, true);
        focusTree_.updateTree(peers, global_.assignment(), global_.treeLeaves(), box(), std::get<0>(scratch));
        focusTree_.updateCounts(keyView, global_.treeLeaves(), global_.nodeCounts(), std::get<0>(scratch));

        reallocate(focusTree_.octreeViewAcc().numLeafNodes + 1, allocGrowthRate_, layout_, layoutAcc_);
        focusTree_.discoverHalos(rawPtr(x), rawPtr(y), rawPtr(z), rawPtr(h), {rawPtr(layoutAcc_), layoutAcc_.size()},
                                 haloSearchExt_, get<0>(scratch), false);
        focusTree_.computeLayout({rawPtr(layoutAcc_), layoutAcc_.size()}, layout_);
        halos_.exchangeRequests(focusTree_.treeLeaves(), focusTree_.assignment(), peers, layout_);

        updateLayout(sorter, keyView, particleKeys, std::tie(x, y, z, h), particleProperties, scratch);
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
        SfcSorter sorter(sfcOrder);

        auto scratch = util::discardLastElement(scratchBuffers);

        auto [exchangeStart, keyView] =
            distribute(sorter, particleKeys, x, y, z, std::tuple_cat(std::tie(h, m), particleProperties), scratch);
        gatherArrays({sorter.getMap() + global_.postExchangeStart(bufDesc_), global_.numAssigned()}, 0,
                     std::tie(x, y, z, h, m), util::reverse(scratch));

        float invThetaEff      = invThetaMinToVec(theta_);
        std::vector<int> peers = findPeersMac(myRank_, global_.assignment(), global_.octreeHost(), box(), invThetaEff);

        if (firstCall_)
        {
            // first rough convergence to avoid computing expansion centers of large nodes with a lot of particles
            focusTree_.converge(box(), keyView, peers, global_.assignment(), global_.treeLeaves(), global_.nodeCounts(),
                                1.0, std::get<0>(scratch));
            focusTree_.updateMinMac(global_.assignment(), 1.0, false);
            int converged = 0, reps = 0;
            while (converged != numRanks_ || reps < 2)
            {
                converged = focusTree_.updateTree(peers, global_.assignment(), global_.treeLeaves(), box(),
                                                  std::get<0>(scratch));
                focusTree_.updateCounts(keyView, global_.treeLeaves(), global_.nodeCounts(), std::get<0>(scratch));
                focusTree_.updateCenters(rawPtr(x), rawPtr(y), rawPtr(z), rawPtr(m), global_.octree(),
                                         std::get<0>(scratch), std::get<1>(scratch));
                focusTree_.updateMacs(global_.assignment(), 1.0 / theta_, false);
                MPI_Allreduce(MPI_IN_PLACE, &converged, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                reps++;
            }
        }

        int fail = 0;
        do
        {
            focusTree_.updateMacs(global_.assignment(), centerDriftTol_ / theta_, false);
            focusTree_.updateTree(peers, global_.assignment(), global_.treeLeaves(), box(), std::get<0>(scratch));
            focusTree_.updateCounts(keyView, global_.treeLeaves(), global_.nodeCounts(), std::get<0>(scratch));
            focusTree_.updateCenters(rawPtr(x), rawPtr(y), rawPtr(z), rawPtr(m), global_.octree(), std::get<0>(scratch),
                                     std::get<1>(scratch));
            focusTree_.updateMacs(global_.assignment(), 1.0 / theta_, false);

            reallocate(focusTree_.octreeViewAcc().numLeafNodes + 1, allocGrowthRate_, layout_, layoutAcc_);
            focusTree_.discoverHalos(rawPtr(x), rawPtr(y), rawPtr(z), rawPtr(h),
                                     {rawPtr(layoutAcc_), layoutAcc_.size()}, haloSearchExt_, get<0>(scratch), true);
            fail = focusTree_.computeLayout({rawPtr(layoutAcc_), layoutAcc_.size()}, layout_);
            MPI_Allreduce(MPI_IN_PLACE, &fail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

            halos_.exchangeRequests(focusTree_.treeLeaves(), focusTree_.assignment(), peers, layout_);

            if (fail)
            {
                centerDriftTol_ += 0.05;
                if (myRank_ == 0) { std::cout << "Increased centerDriftTol to " << centerDriftTol_ << std::endl; }
            }
        } while (fail);

        // diagnostics(keyView.size(), peers);

        updateLayout(sorter, keyView, particleKeys, std::tie(x, y, z, h, m), particleProperties, scratch);
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
        std::apply([this](auto&... arrays) { this->checkSizesEqual(this->prevBufDesc_.size, arrays...); }, arrays);

        LocalIndex exSize =
            domain_exchange::exchangeBufferSize(prevBufDesc_, global_.numPresent(), global_.numAssigned());
        lowMemReallocate(exSize, allocGrowthRate_, arrays, {});

        BufferDescription exDesc{prevBufDesc_.start, prevBufDesc_.end, exSize};
        auto envelope = domain_exchange::assignedEnvelope(exDesc, global_.numAssigned() - global_.numPresent());

        auto* ord = reinterpret_cast<LocalIndex*>(rawPtr(ordering)) + envelope[0];
        std::vector<LocalIndex> orderingCpu;
        if constexpr (HaveGpu<Accelerator>{})
        {
            static_assert(IsDeviceVector<OVec>{}, "Need ordering on GPU for GPU-accelerated domain");
            orderingCpu.resize(envelope[1] - envelope[0]);
            memcpyD2H(ord, orderingCpu.size(), orderingCpu.data());
            ord = orderingCpu.data();
        }

        std::apply([exDesc, ord, &sendBuffer, &receiveBuffer, this](auto&... a)
                   { global_.redoExchange(exDesc, ord, sendBuffer, receiveBuffer, rawPtr(a)...); }, arrays);

        lowMemReallocate(bufDesc_.size, allocGrowthRate_, arrays, std::tie(sendBuffer, receiveBuffer));
        gatherArrays({ord + global_.numSendDown(), global_.numAssigned()}, bufDesc_.start, arrays,
                     std::tie(sendBuffer, receiveBuffer));
    }

    //! @brief repeat the halo exchange pattern from the previous sync operation for a different set of arrays
    template<class... Vectors, class SendBuffer, class ReceiveBuffer>
    void exchangeHalos(std::tuple<Vectors&...> arrays, SendBuffer& sendBuffer, ReceiveBuffer& receiveBuffer) const
    {
        std::apply([this](auto&... arrays) { this->checkSizesEqual(this->bufDesc_.size, arrays...); }, arrays);
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
    //! @brief read only visibility of the global octree in traversible layout
    OctreeView<const KeyType> globalTree() const { return global_.octree(); }
    //! @brief read only visibility of the focused octree
    const FocusedOctree<KeyType, T, Accelerator>& focusTree() const { return focusTree_; }
    //! @brief the index of the first locally assigned cell in focusTree()
    TreeNodeIndex startCell() const { return focusTree_.assignment()[myRank_].start(); }
    //! @brief the index of the last locally assigned cell in focusTree()
    TreeNodeIndex endCell() const { return focusTree_.assignment()[myRank_].end(); }
    //! @brief particle offsets of each focus tree leaf cell
    std::span<const LocalIndex> layout() const { return {rawPtr(layoutAcc_), layoutAcc_.size()}; }
    //! @brief return the coordinate bounding box from the previous sync call
    const Box<T>& box() const { return global_.box(); }

    KeyType assignmentStart() const { return global_.assignment()[myRank_]; }

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
                ft.parents,
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

        auto valueTypeCheck = [](auto m, auto s)
        {
            constexpr int numScratchBuffers = std::tuple_size_v<std::decay_t<decltype(tup)>>;
            static_assert(m < numScratchBuffers || s < numScratchBuffers,
                          "one of the conserved fields has a value_type bigger than the value_types of available "
                          "scratch buffers");
        };
        util::for_each_tuple(valueTypeCheck, matches, smaller);
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
                      std::span<const KeyType> keyView,
                      KeyVec& keys,
                      std::tuple<Arrays1&...> orderedBuffers,
                      std::tuple<Arrays2&...> unorderedBuffers,
                      std::tuple<Arrays3&...> scratchBuffers)
    {
        auto myRange = focusTree_.assignment()[myRank_];
        BufferDescription newBufDesc{layout_[myRange.start()], layout_[myRange.end()], layout_.back()};

        lowMemReallocate(newBufDesc.size, allocGrowthRate_, std::tuple_cat(orderedBuffers, unorderedBuffers),
                         scratchBuffers);

        // copy or H2D upload
        layoutAcc_ = layout_;

        // re-locate particle SFC keys
        constexpr int i = util::FindIndex<KeyVec&, std::tuple<Arrays3&...>, SmallerElementSize>{};
        constexpr int j = (i >= sizeof...(Arrays3)) ? 0 : i;
        auto& swapSpace = std::get<j>(scratchBuffers);
        size_t origSize = reallocateBytes(swapSpace, keyView.size() * sizeof(KeyType), allocGrowthRate_);
        auto* swapPtr   = reinterpret_cast<KeyType*>(swapSpace.data());
        copy_n<HaveGpu<Accelerator>{}>(keyView.data(), keyView.size(), swapPtr);
        reallocate(keys, newBufDesc.size, allocGrowthRate_);
        fill<HaveGpu<Accelerator>{}>(rawPtr(keys) + bufDesc_.size, rawPtr(keys) + newBufDesc.size, KeyType(0));
        copy_n<HaveGpu<Accelerator>{}>(swapPtr, keyView.size(), rawPtr(keys) + newBufDesc.start);
        reallocate(swapSpace, origSize, 1.0);

        // relocate ordered buffer contents from offset 0 to offset newBufDesc.start
        auto relocate =
            [size = keyView.size(), dest = newBufDesc.start, scratch = util::reverse(scratchBuffers)](auto& array)
        {
            static_assert(util::Contains<decltype(array), std::tuple<Arrays3&...>>{}, "No suitable scratch buffer");
            auto& swapSpace = util::pickType<decltype(array)>(scratch);
            copy_n<IsDeviceVector<std::decay_t<decltype(array)>>{}>(rawPtr(array), size, rawPtr(swapSpace) + dest);
            swap(array, swapSpace);
        };
        util::for_each_tuple(relocate, orderedBuffers);

        // reorder the unordered buffers
        gatherArrays({sorter.getMap() + global_.postExchangeStart(bufDesc_), global_.numAssigned()}, newBufDesc.start,
                     unorderedBuffers, util::reverse(scratchBuffers));

        // newBufDesc is now the valid buffer description
        prevBufDesc_ = bufDesc_;
        bufDesc_     = newBufDesc;
    }

    void diagnostics(size_t assignedSize, std::span<int> peers)
    {
        auto focusAssignment = focusTree_.assignment();
        auto focusTree       = focusTree_.treeLeaves();
        auto globalTree      = global_.treeLeaves();

        std::vector<KeyType> globalTreeBackingBuffer;
        if constexpr (cstone::HaveGpu<Accelerator>{})
        {
            globalTreeBackingBuffer.resize(globalTree.size());
            memcpyD2H(globalTree.data(), globalTree.size(), globalTreeBackingBuffer.data());
            globalTree = std::span(globalTreeBackingBuffer);
        }

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
                    TreeNodeIndex gi = findNodeAbove(globalTree.data(), globalTree.size(), fnstart);
                    if (!(gi < nNodes(globalTree) && globalTree[gi] == fnstart && globalTree[gi + 1] <= fnend))
                    {
                        numFocusTruePeer++;
                    }
                }
            }
        }

        int numFlags = std::count(focusTree_.haloFlags().begin(), focusTree_.haloFlags().end(), 1);
        for (int i = 0; i < numRanks_; ++i)
        {
            if (i == myRank_)
            {
                std::cout << "rank " << i << " " << assignedSize << " " << layout_.back()
                          << " focus h/true/peers/loc/tot: " << numFlags << "/" << numFocusTruePeer << "/"
                          << numFocusPeers << "/" << focusAssignment[myRank_].count() << "/"
                          << focusTree_.haloFlags().size() << " peers: [" << peers.size() << "] ";
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

    GlobalAssignment<KeyType, T, Accelerator> global_;

    Halos<KeyType, Accelerator> halos_{myRank_};

    bool firstCall_{true};
};

} // namespace cstone
