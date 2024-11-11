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
 * @brief Utility functions to resolve names of particle fields to pointers
 *
 * C++17 compatible for use with Simulation Datasets
 */

#pragma once

#include <stdexcept>
#include <string_view>
#include <vector>

namespace cstone
{

//! @brief compile-time index look-up of a string literal in a list of strings
template<class Array>
constexpr size_t getFieldIndex(std::string_view field, const Array& fieldNames)
{
    for (size_t i = 0; i < fieldNames.size(); ++i)
    {
        if (field == fieldNames[i]) { return i; }
    }
    return fieldNames.size();
}

/*! @brief Look up indices of a (runtime-variable) number of field names
 *
 * @tparam     Array
 * @param[in]  subsetNames  array of strings of field names to look up in @p allNames
 * @param[in]  allNames     array of strings with names of all fields
 * @return                  the indices of @p subsetNames in @p allNames
 */
template<class Array>
std::vector<int> fieldStringsToInt(const std::vector<std::string>& subsetNames, const Array& allNames)
{
    std::vector<int> subsetIndices(subsetNames.size());
    for (size_t i = 0; i < subsetNames.size(); ++i)
    {
        subsetIndices[i] = getFieldIndex(subsetNames[i], allNames);
        if (size_t(subsetIndices[i]) == allNames.size())
        {
            throw std::runtime_error("Field not found: " + subsetNames[i]);
        }
    }
    return subsetIndices;
}

} // namespace cstone
