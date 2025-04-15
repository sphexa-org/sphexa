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
