/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * @brief Target particle group configuration
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/traversal/groups_gpu.h"
#include "sph/sph_gpu.hpp"

namespace sph
{

template<class Dataset>
void computeSpatialGroups(cstone::LocalIndex startIndex, cstone::LocalIndex endIndex, Dataset& d,
                          const cstone::Box<typename Dataset::RealType>& box, GroupData<cstone::GpuTag>& groups)
{
    float tolFactor = 2.0f;
    cstone::computeGroupSplits(startIndex, endIndex, rawPtr(d.devData.x), rawPtr(d.devData.y), rawPtr(d.devData.z),
                               rawPtr(d.devData.h), d.treeView.leaves, d.treeView.numLeafNodes, d.treeView.layout, box,
                               nsGroupSize(), tolFactor, d.devData.traversalStack, groups.data);

    groups.firstBody  = startIndex;
    groups.lastBody   = endIndex;
    groups.numGroups  = groups.data.size() - 1;
    groups.groupStart = rawPtr(groups.data);
    groups.groupEnd   = rawPtr(groups.data) + 1;
}

//! @brief Compute spatial (=SFC-consecutive) groups of particles with compact bounding boxes
template<typename Tc, class Dataset>
void computeGroups(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<Tc>& box,
                   GroupData<typename Dataset::AcceleratorType>& groups)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        computeSpatialGroups(startIndex, endIndex, d, box, groups);
    }
    else
    {
        groups.firstBody  = startIndex;
        groups.lastBody   = endIndex;
        groups.numGroups  = 1;
        groups.groupStart = &groups.firstBody;
        groups.groupEnd   = &groups.lastBody;
    }
}

//! @brief extract the specified subgroup [first:last] indexed through @p index from @p grp into @p outGroup
template<class Accelerator>
inline void extractGroupGpu(const GroupView& grp, const cstone::LocalIndex* indices, cstone::LocalIndex first,
                            cstone::LocalIndex last, GroupData<Accelerator>& out)
{
    auto numOutGroups = last - first;
    reallocate(out.data, 2 * numOutGroups, 1.01);

    out.firstBody  = 0;
    out.lastBody   = 0;
    out.numGroups  = numOutGroups;
    out.groupStart = rawPtr(out.data);
    out.groupEnd   = rawPtr(out.data) + numOutGroups;

    if (numOutGroups == 0) { return; }
    cstone::gatherGpu(indices + first, numOutGroups, grp.groupStart, out.groupStart);
    cstone::gatherGpu(indices + first, numOutGroups, grp.groupEnd, out.groupEnd);
}

//! @brief return a new GroupView that corresponds to a slice [first:last] of the input group @p grp
inline GroupView makeSlicedView(const GroupView& grp, cstone::LocalIndex first, cstone::LocalIndex last)
{
    GroupView ret = grp;
    ret.numGroups = last - first;
    ret.groupStart += first;
    ret.groupEnd += first;
    return ret;
}

} // namespace sph
