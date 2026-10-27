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

#include "cstone/primitives/primitives_gpu.h"
#include "cstone/traversal/groups.hpp"
#include "sph/sph_gpu.hpp"

namespace sph
{

//! @brief Compute spatial (=SFC-consecutive) groups of particles: all groups of the range are active
/*!
 * The returned group data does not store an index list: the GroupView obtained through view() describes
 * all consecutive groups of cstone::groupSize particles in [startIndex, endIndex).
 */
template<cstone::execution::Policy Exec>
void computeGroups(size_t startIndex, size_t endIndex, GroupData<Exec>& groups)
{
    groups.firstBody = startIndex;
    groups.lastBody  = endIndex;
}

//! @brief extract the specified subgroup [first:last] indexed through @p indices from @p grp into @p out
/*! the extracted view keeps the original firstBody/lastBody of @p grp, while out.activeGroups stores the
 *  selected group indices (a plain copy of indices[first:last])
 */
template<cstone::execution::Policy Exec>
inline void extractGroupGpu(const GroupView& grp, const cstone::LocalIndex* indices, cstone::LocalIndex first,
                            cstone::LocalIndex last, GroupData<Exec>& out)
{
    auto numOutGroups = last - first;
    reallocate(out.data, numOutGroups, 1.01);

    out.firstBody       = grp.firstBody;
    out.lastBody        = grp.lastBody;
    out.numActiveGroups = numOutGroups;
    out.activeGroups    = rawPtr(out.data);

    if (numOutGroups == 0) { return; }
    cstone::memcpyD2DAsync(cstone::execution::gpuDefaultStream, indices + first, numOutGroups, out.activeGroups);
}

//! @brief return a new GroupView that corresponds to a slice [first:last] of the input group @p grp
inline GroupView makeSlicedView(const GroupView& grp, cstone::LocalIndex first, cstone::LocalIndex last)
{
    GroupView ret         = grp;
    ret.activeGroups     += first;
    ret.numActiveGroups   = last - first;
    return ret;
}

} // namespace sph
