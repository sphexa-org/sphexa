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
 * @brief  Build a tree for Ryoanji with the cornerstone framework
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/thrust_util.cuh"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/sfc/sfc_gpu.h"
#include "cstone/tree/octree_gpu.h"
#include "cstone/tree/update_gpu.cuh"

#include "ryoanji/nbody/types.h"

namespace ryoanji
{

template<class KeyType>
class TreeBuilder
{
public:
    //! @brief initialize with the desired maximum particles per leaf cell
    TreeBuilder(unsigned ncrit) : bucketSize_(ncrit) {}

    /*! @brief construct an octree from body coordinates
     *
     * @tparam        T           float or double
     * @param[inout]  x           body x-coordinates, will be sorted in SFC-order
     * @param[inout]  y           body y-coordinates, will be sorted in SFC-order
     * @param[inout]  z           body z-coordinates, will be sorted in SFC-order
     * @param[in]     numBodies   number of bodies in @p x,y,z
     * @param[in]     box         the coordinate bounding box
     * @return                    the total number of cells in the constructed octree
     *
     * Note: x,y,z arrays will be sorted in SFC order to match be consistent with the cell body offsets of the tree
     */
    template<class T>
    int update(T* x, T* y, T* z, size_t numBodies, const cstone::Box<T>& box)
    {
        thrust::device_vector<KeyType> d_keys(numBodies), d_keys_tmp(numBodies);
        thrust::device_vector<int>     d_ordering(numBodies), d_values_tmp(numBodies);
        thrust::device_vector<T>       tmp(numBodies);

        uint64_t                    tempStorageEle = cstone::sortByKeyTempStorage<KeyType, LocalIndex>(numBodies);
        thrust::device_vector<char> cubTmpStorage(tempStorageEle);

        cstone::computeSfcKeysGpu(x, y, z, cstone::sfcKindPointer(rawPtr(d_keys)), numBodies, box);

        cstone::sequenceGpu(rawPtr(d_ordering), d_ordering.size(), 0);
        cstone::sortByKeyGpu(rawPtr(d_keys), rawPtr(d_keys) + d_keys.size(), rawPtr(d_ordering), rawPtr(d_keys_tmp),
                             rawPtr(d_values_tmp), rawPtr(cubTmpStorage), tempStorageEle);

        thrust::gather(thrust::device, d_ordering.begin(), d_ordering.end(), x, tmp.begin());
        thrust::copy(tmp.begin(), tmp.end(), x);
        thrust::gather(thrust::device, d_ordering.begin(), d_ordering.end(), y, tmp.begin());
        thrust::copy(tmp.begin(), tmp.end(), y);
        thrust::gather(thrust::device, d_ordering.begin(), d_ordering.end(), z, tmp.begin());
        thrust::copy(tmp.begin(), tmp.end(), z);

        if (d_tree_.size() == 0)
        {
            // initial guess on first call. use previous tree as guess on subsequent calls
            d_tree_   = std::vector<KeyType>{0, cstone::nodeRange<KeyType>(0)};
            d_counts_ = std::vector<unsigned>{unsigned(numBodies)};
        }

        while (!cstone::updateOctreeGpu(rawPtr(d_keys), rawPtr(d_keys) + d_keys.size(), bucketSize_, d_tree_, d_counts_,
                                        tmpTree_, workArray_))
            ;

        octreeGpuData_.resize(cstone::nNodes(d_tree_));
        cstone::buildOctreeGpu(rawPtr(d_tree_), octreeGpuData_.data());

        d_layout_.resize(d_counts_.size() + 1);
        cstone::fillGpu(rawPtr(d_layout_), rawPtr(d_layout_) + 1, LocalIndex(0));
        cstone::inclusiveScanGpu(rawPtr(d_counts_), rawPtr(d_counts_) + d_counts_.size(), rawPtr(d_layout_) + 1);

        levelRange_host_ = toHost(octreeGpuData_.levelRange);
        return octreeGpuData_.numInternalNodes + octreeGpuData_.numLeafNodes;
    }

    const LocalIndex*    layout() const { return rawPtr(d_layout_); }
    const KeyType*       nodeKeys() const { return rawPtr(octreeGpuData_.prefixes); }
    const TreeNodeIndex* childOffsets() const { return rawPtr(octreeGpuData_.childOffsets); }
    const TreeNodeIndex* leafToInternal() const
    {
        return cstone::leafToInternal(octreeGpuData_).data();
    }
    const TreeNodeIndex* internalToLeaf() const { return rawPtr(octreeGpuData_.internalToLeaf); }
    //! @brief return host-resident octree level cell ranges
    const TreeNodeIndex* levelRange() const { return levelRange_host_.data(); }

    TreeNodeIndex numLeafNodes() const { return octreeGpuData_.numLeafNodes; }
    unsigned      maxTreeLevel() const { return cstone::maxTreeLevel<KeyType>{}; }

private:
    unsigned bucketSize_;

    thrust::device_vector<KeyType>  d_tree_;
    thrust::device_vector<unsigned> d_counts_;

    thrust::device_vector<KeyType>               tmpTree_;
    thrust::device_vector<cstone::TreeNodeIndex> workArray_;

    cstone::OctreeData<KeyType, cstone::GpuTag> octreeGpuData_;
    thrust::device_vector<cstone::LocalIndex>   d_layout_;

    std::vector<TreeNodeIndex> levelRange_host_;
};

} // namespace ryoanji
