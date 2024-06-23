
/*! @file
 * @brief CPU/GPU wrapper
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/tree/accel_switch.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/primitives/primitives_gpu.h"

namespace cstone
{

template<bool useGpu, class IndexType, class ValueType>
void gatherAcc(gsl::span<const IndexType> ordering, const ValueType* source, ValueType* destination)
{
    if constexpr (useGpu) { gatherGpu(ordering, source, destination); }
    else { gather(ordering, source, destination); }
}

template<bool useGpu, class IndexType, class ValueType>
void scatterAcc(gsl::span<const IndexType> ordering, const ValueType* source, ValueType* destination)
{
    if constexpr (useGpu) { scatterGpu(ordering, source, destination); }
    else { scatter(ordering, source, destination); }
}

} // namespace cstone
