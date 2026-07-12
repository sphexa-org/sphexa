/*
 * MIT License
 *
 * Copyright (c) 2025 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * @brief Id tagging setup utilities
 *
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 */

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string_view>
#include <vector>

#include "ifile_io.hpp"
#include "id_tag_utils.hpp"

namespace sphexa
{
    /*! @brief Concatenate two constexpr arrays of the same element type
     */
    template<class T, std::size_t N, std::size_t M>
    constexpr std::array<T, N + M> concatArrays(const std::array<T, N>& a, const std::array<T, M>& b)
    {
        std::array<T, N + M> result{};
        std::size_t          i = 0;
        for (const auto& e : a) { result[i++] = e; }
        for (const auto& e : b) { result[i++] = e; }
        return result;
    }

    /*! @brief Id tag (selection input) file attribute names, positionally accessed in isim_init
     */
    inline constexpr std::array<const char*, 4> idTagAttributes = {
        "id_selection_spheres",
        "id_selection_spheres_group_ids",
        "id_selection_list",
        "id_selection_list_group_ids"
    };

    /*! @brief Id tag output (subset) file attribute names
     */
    inline constexpr std::array<const char*, 4> idTagOutputAttributes = {
        "o_subset",
        "w_subset",
        "wextra_subset",
        "f_subset"
    };

    /*! @brief All id tagging file attributes (selection inputs + subset outputs),
     *         excluded from the generic InitSettings read
     */
    inline constexpr auto idTagFsStrings = concatArrays(idTagAttributes, idTagOutputAttributes);

    /*! @brief Id tagging output parameters
     */
    struct IdTaggingOutputSetup
    {
        // Tagged id output file name
        std::string outFile;
        // Write frequency (iterations or time)
        std::string writeFreqStr = "0";
        // Output fields
        std::vector<std::string> outputFields;
        // Extra writes steps/times
        std::vector<std::string> writeExtra;
    };

    /*! @brief Id tagging output handling parameter retrieval
     *
     * @param[in]  initCond         initial condition
     * @param[in]  reader           parameter file reader
     * @param[in]  outputFileSuffix suffix to append to output file name
     *
     * @return IdTaggingOutputSetup if id tagging output is requested, std::nullopt otherwise
     */
    std::optional<IdTaggingOutputSetup> readFileTaggingOutputAttributes(const std::string& initCond, IFileReader* reader,
                                                                         const std::string& outputFileSuffix);

}
