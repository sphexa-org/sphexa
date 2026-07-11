//
// Created by Noah Kubli on 11.07.2026.
//

#include <cstdint>
#include <span>

#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include "sph/id_layout.hpp"
#include "iad_regularization_stat.hpp"

namespace sph
{

std::size_t countAndCleanRegTagGPU(std::span<std::uint64_t> id, std::size_t firstIndex, std::size_t lastIndex,
                                   bool clear)
{
    auto first = thrust::device_pointer_cast(id.data() + firstIndex);
    auto last  = thrust::device_pointer_cast(id.data() + lastIndex);

    auto              check_mask = [] __device__(std::uint64_t value) { return (value & iadRegularizationMask) != 0; };
    const std::size_t count      = thrust::count_if(thrust::device, first, last, check_mask);

    if (clear)
    {
        auto clear_mask = [] __device__(std::uint64_t value) { return value & ~iadRegularizationMask; };
        thrust::transform(thrust::device, first, last, first, clear_mask);
    }
    return count;
}

} // namespace sph
