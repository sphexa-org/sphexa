/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
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
