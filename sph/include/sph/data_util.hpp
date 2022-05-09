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
 */

#pragma once

#include <vector>
#include <variant>

#include "traits.hpp"

namespace sphexa
{

/*! @brief look up indices of field names
 *
 * @tparam     Array
 * @param[in]  allNames     array of strings with names of all fields
 * @param[in]  subsetNames  array of strings of field names to look up in @p allNames
 * @return                  the indices of @p subsetNames in @p allNames
 */
template<class Array>
std::vector<int> fieldStringsToInt(const Array& allNames, const std::vector<std::string>& subsetNames)
{
    std::vector<int> subsetIndices;
    subsetIndices.reserve(subsetNames.size());
    for (const auto& field : subsetNames)
    {
        auto it = std::find(allNames.begin(), allNames.end(), field);
        if (it == allNames.end()) { throw std::runtime_error("Field " + field + " does not exist\n"); }

        size_t fieldIndex = it - allNames.begin();
        subsetIndices.push_back(fieldIndex);
    }
    return subsetIndices;
}

//! @brief extract a vector of pointers to particle fields for file output
template<class Dataset>
auto getOutputArrays(Dataset& dataset)
{
    auto fieldPointers = dataset.data();
    using FieldType    = std::variant<float*, double*, int*, unsigned*, uint64_t*>;

    std::vector<FieldType> outputFields;
    outputFields.reserve(dataset.outputFieldIndices.size());

    for (int i : dataset.outputFieldIndices)
    {
        std::visit([&outputFields](auto& arg) { outputFields.push_back(arg->data()); }, fieldPointers[i]);
    }
    return outputFields;
}

/*! @brief resizes all active particles fields of @p d to the specified size
 *
 * Important Note: this only resizes the fields that are listed either as conserved or dependent.
 * The conserved/dependent list may be set at runtime, depending on the need of the simulation!
 */
template<class Dataset>
void resize(Dataset& d, size_t size)
{
    double growthRate = 1.05;
    auto   data_      = d.data();

    for (int i : d.conservedFields)
    {
        std::visit([size, growthRate](auto& arg) { reallocate(*arg, size, growthRate); }, data_[i]);
    }
    for (int i : d.dependentFields)
    {
        std::visit([size, growthRate](auto& arg) { reallocate(*arg, size, growthRate); }, data_[i]);
    }

    d.devPtrs.resize(size);
}

//! @brief resizes the neighbors list, only used in the CPU version
template<class Dataset>
void resizeNeighbors(Dataset& d, size_t size)
{
    double growthRate = 1.05;
    //! If we have a GPU, neighbors are calculated on-the-fly, so we don't need space to store them
    reallocate(d.neighbors, HaveGpu<typename Dataset::AcceleratorType>{} ? 0 : size, growthRate);
}

/*! @brief Determine storage location for field with index @p what
 *
 * @tparam FieldPointers
 * @param  what             the field index to store
 * @param  outputFields     indices of output fields
 * @param  conservedFields  indices of conserved fields
 * @param  dependentFields  indices of dependent fields
 * @param  fieldPointers    the array of variants returned by ParticlesData::data()
 * @return                  @p what if @p what is either conserved or dependent
 *                          otherwise the first field in @p dependentFields whose type matches the type of @p what
 *                          and which is not listed in @p outputFields
 *
 * Field indices are relative to the names table in ParticlesData::fieldNames.
 *
 * Example usage: With volume elements, rho is not an active field. Instead, rho can be computed as rho = kx * m / xm.
 * When output of rho is requested, we look for a field which we can fill with the values of kx * m / xm.
 * Since outputs happen an the end of the iteration, we may use any dependent field (not conserved between iterations)
 * which is not itself contained in the list of output fields.
 */
template<class FieldPointers>
int findFieldIdx(int what, gsl::span<const int> outputFields, gsl::span<const int> conservedFields,
                 gsl::span<const int> dependentFields, const FieldPointers& fieldPointers)
{
    bool fieldIsActive = (std::find(conservedFields.begin(), conservedFields.end(), what) != conservedFields.end()) ||
                         (std::find(dependentFields.begin(), dependentFields.end(), what) != dependentFields.end());

    if (fieldIsActive) { return what; }

    for (int field : dependentFields)
    {
        bool fieldIsNotOutput = std::find(outputFields.begin(), outputFields.end(), field) == outputFields.end();

        auto checkTypesMatch = [](const auto* varPtr1, const auto* varPtr2)
        {
            using VectorType1 = std::decay_t<decltype(*varPtr1)>;
            using VectorType2 = std::decay_t<decltype(*varPtr2)>;

            return std::is_same_v<typename VectorType1::value_type, typename VectorType2::value_type>;
        };

        bool typesMatch = std::visit(checkTypesMatch, fieldPointers[what], fieldPointers[field]);

        if (fieldIsNotOutput && typesMatch) { return field; }
    }

    return fieldPointers.size();
}

} // namespace sphexa
