//
// Created by Noah Kubli on 11.07.2026.
//

#pragma once

#include <cstdint>
#include <limits>

namespace sph
{
struct IDLayout
{
    /*! @brief Number of bits used for tagging information storage
     */
    inline constexpr static std::uint64_t tagNumBits = 10;

    /*! @brief First tagging bit position
     */
    inline constexpr static std::uint64_t taggingMaskStartingBit =
        std::numeric_limits<std::uint64_t>::digits - tagNumBits;
    /*! @brief Additional reserved bit for IAD regularization statistics
     */
    inline constexpr static std::uint64_t iadRegBit = taggingMaskStartingBit - 1;
};
} // namespace sph
