/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief  Simple CSV output for benchmark results
 *
 * @author Felix Thaler <thaler@cscs.ch>
 */

#pragma once

#include <fstream>
#include <map>
#include <ostream>
#include <string>
#include <vector>

namespace detail
{

template<class T>
void saveCsvImpl(std::ostream& out, const std::map<std::string, std::vector<T>>& data)
{
    std::size_t numRows = 0;
    {
        bool first = true;
        for (const auto& [name, vec] : data)
        {
            if (first)
                first = false;
            else
                out << ",";
            out << name;
            numRows = std::max(numRows, vec.size());
        }
    }
    out << "\n";
    for (std::size_t row = 0; row < numRows; ++row)
    {
        bool first = true;
        for (const auto& [_, vec] : data)
        {
            if (first)
                first = false;
            else
                out << ",";
            if (row < vec.size()) out << vec[row];
        }
        out << "\n";
    }
}
} // namespace detail

template<class Path, class T>
void saveCsv(Path&& filename, const std::map<std::string, std::vector<T>>& data)
{
    if (data.empty()) throw std::runtime_error("ERROR writing CSV: no data passed!");

    if constexpr (std::is_base_of_v<std::ostream, std::remove_cvref_t<Path>>)
    {
        detail::saveCsvImpl(std::forward<Path>(filename), data);
    }
    else
    {
        std::ofstream file(std::forward<Path>(filename));
        if (!file) throw std::runtime_error("ERROR writing CSV: could not open file for writing!");
        detail::saveCsvImpl(file, data);
    }
}
