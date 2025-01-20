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
 * @brief Min-reduction to determine global timestep
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include <mpi.h>

#include "cstone/primitives/math.hpp"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/util/pack_buffers.hpp"
#include "sph/kernels.hpp"
#include "sph/timestep.h"

namespace sph
{

//! @brief compute Divv-limited timestep for each group when block time-steps are active
template<class Dataset>
void groupDivvTimestep(const GroupView& grp, float* groupDt, const Dataset& d)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        groupDivvTimestepGpu(d.Krho, grp, rawPtr(d.devData.divv), groupDt);
    }
}

//! @brief compute acceleration-limited timestep for each group when block time-steps are active
template<class Dataset>
void groupAccTimestep(const GroupView& grp, float* groupDt, const Dataset& d)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        groupAccTimestepGpu(d.etaAcc * std::sqrt(d.eps), grp, rawPtr(d.devData.ax), rawPtr(d.devData.ay),
                            rawPtr(d.devData.az), groupDt);
    }
}

//! @brief sort groupDt, keeping track of the ordering
template<class AccVec>
void sortGroupDt(float* groupDt, cstone::LocalIndex* groupIndices, cstone::LocalIndex numGroups, AccVec& scratch)
{
    using cstone::LocalIndex;
    size_t oldSize = scratch.size();

    // radix sort temp storage as multiples of float
    size_t tempElem       = cstone::iceil(cstone::sortByKeyTempStorage<float, LocalIndex>(numGroups), sizeof(float));
    size_t numElements[3] = {numGroups, numGroups * sizeof(LocalIndex) / sizeof(float), tempElem};

    auto  buffers     = util::packAllocBuffer<float>(scratch, {numElements, 3}, 128);
    auto* valueBuf    = reinterpret_cast<LocalIndex*>(buffers[1].data());
    void* tempStorage = buffers[2].data();
    cstone::sequenceGpu(groupIndices, numGroups, 0u);
    cstone::sortByKeyGpu(groupDt, groupDt + numGroups, groupIndices, buffers[0].data(), valueBuf, tempStorage,
                         tempElem * sizeof(float));
    reallocate(oldSize, 1.0, scratch);
};

//! @brief return the local minimum timestep and the biggest timestep of the fastest fraction of particles
inline auto timestepRangeGpu(const float* groupDt, cstone::LocalIndex numGroups, float fastFraction)
{
    std::array<float, 2> minGroupDt;
    memcpyD2H(groupDt, 1, minGroupDt.data());
    memcpyD2H(groupDt + cstone::LocalIndex(fastFraction * numGroups), 1, minGroupDt.data() + 1);
    return minGroupDt;
}

template<class AccVec>
auto computeMinTimestep(float* groupDt, LocalIndex* groupIndices, LocalIndex numGroups, LocalIndex numGroupsTot,
                        AccVec& scratch)
{
    float                fastFraction = 0.4;
    std::array<float, 2> minGroupDt;
    if constexpr (IsDeviceVector<AccVec>{})
    {
        sortGroupDt(groupDt, groupIndices, numGroups, scratch);
        cstone::sequenceGpu(groupIndices + numGroups, numGroupsTot - numGroups, numGroups);
        minGroupDt = timestepRangeGpu(groupDt, numGroups, fastFraction);
    }

    std::array<float, 2> minDtGlobal;
    mpiAllreduce(minGroupDt.data(), minDtGlobal.data(), minGroupDt.size(), MPI_MIN);
    return minDtGlobal;
}

/*! @brief find ranges of 2*minDt, 4*minDt, 8*minDt
 *
 * @tparam useGpu
 * @param minDt       global minimum time step
 * @param groupDt     sorted timesteps per groups
 * @param numGroups   length of @p groupDt
 * @param numRungs    number of power-of-2 multiples of @p minDt to search
 * @return            indices groupDt elements >2 * minDt, >4 * minDt, etc
 */
template<bool useGpu>
auto findRungRanges(float minDt, const float* groupDt, LocalIndex numGroups, int numRungs)
{
    std::array<LocalIndex, Timestep::maxNumRungs + 1> rungRanges{0};
    std::fill(rungRanges.begin() + 1, rungRanges.end(), numGroups);
    for (int rung = 1; rung < numRungs; ++rung)
    {
        float maxDtRung = (1 << rung) * minDt;
        if constexpr (useGpu)
            rungRanges[rung] = cstone::lowerBoundGpu(groupDt, groupDt + numGroups, maxDtRung);
        else
            rungRanges[rung] = std::lower_bound(groupDt, groupDt + numGroups, maxDtRung) - groupDt;
    }
    return rungRanges;
}

//! @brief Determine timestep rungs
template<class AccVec>
Timestep rungTimestep(float* groupDt, LocalIndex* groupIndices, LocalIndex numGroups, float maxDt, AccVec& scratch)
{
    auto minDtGlobal = computeMinTimestep(groupDt, groupIndices, numGroups, numGroups, scratch);
    int  numRungs    = std::min(int(log2(minDtGlobal[1] / minDtGlobal[0])) + 1, Timestep::maxNumRungs);
    auto rungRanges  = findRungRanges<IsDeviceVector<AccVec>{}>(minDtGlobal[0], groupDt, numGroups, numRungs);

    minDtGlobal[0]   = std::min(maxDt, minDtGlobal[0]);
    float    totalDt = minDtGlobal[0] * (1 << numRungs);
    Timestep ret{.nextDt = minDtGlobal[0], .totDt = totalDt, .numRungs = numRungs, .rungRanges = rungRanges};
    std::fill(ret.dt_drift.begin(), ret.dt_drift.end(), 0);
    return ret;
}

template<class AccVec>
auto minimumGroupDt(Timestep ts, float* groupDt, LocalIndex* groupIndices, LocalIndex numGroups, AccVec& scratch)
{
    float minDtGlobal = computeMinTimestep(groupDt, groupIndices, numGroups, ts.rungRanges.back(), scratch)[0];
    auto  rungRanges = findRungRanges<IsDeviceVector<AccVec>{}>(minDtGlobal, groupDt, numGroups, Timestep::maxNumRungs);

    float timeLeft     = ts.totDt - ts.elapsedDt;
    int   substepsLeft = (1 << ts.numRungs) - ts.substep;

    return std::make_tuple(std::min(minDtGlobal, timeLeft / substepsLeft), rungRanges);
}

} // namespace sph
