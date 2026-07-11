//
// Created by Noah Kubli on 11.07.2026.
//

#pragma once

#include <cstdint>
#include <span>

#include "cstone/cuda/annotation.hpp"
#include "sph/id_layout.hpp"

namespace sph
{

inline constexpr std::uint64_t iadRegularizationMask = std::uint64_t{1} << sph::IDLayout::iadRegBit;

/*! @brief Set the error bit of id
 * @return tagged id
 */
HOST_DEVICE_FUN inline std::uint64_t setRegularizationTag(bool value, std::uint64_t id)
{
    if (value) { return id | iadRegularizationMask; }
    else { return id & ~iadRegularizationMask; }
}

/*! @brief count number of toggled error bits
 *  @param[in] id
 *  @param[in] firstIndex
 *  @param[in] lastIndex
 *  @param[in] clear        set error bits to 0 after counting
 *  @return number of error bits
 */
std::size_t countAndCleanRegTagGPU(std::span<std::uint64_t> id, std::size_t firstIndex, std::size_t lastIndex,
                                   bool clear);

inline std::size_t countAndCleanRegTag(std::span<std::uint64_t> id, std::size_t firstIndex, std::size_t lastIndex,
                                       bool clear)
{
    std::size_t count{};
#pragma omp parallel for schedule(static) reduction(+ : count)
    for (std::size_t i = firstIndex; i < lastIndex; i++)
    {
        if (bool(id[i] & iadRegularizationMask)) { count++; }
        if (clear) { id[i] &= ~iadRegularizationMask; }
    }
    return count;
}
} // namespace sph
