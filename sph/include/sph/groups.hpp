/*! @file
 * @brief Target particle group configuration
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/primitives/primitives_gpu.h"
#include "sph/sph_gpu.hpp"

namespace sph
{

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
