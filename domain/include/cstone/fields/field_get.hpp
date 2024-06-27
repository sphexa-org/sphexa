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
 * @brief Utility functions to use compile-time strings as arguments to get on tuples
 *
 * Needs C++20 structural types
 */

#pragma once

#include "cstone/fields/data_util.hpp"
#include "cstone/primitives/accel_switch.hpp"
#include "cstone/util/constexpr_string.hpp"
#include "cstone/util/value_list.hpp"

namespace cstone
{

template<class Dataset, class Tuple, util::StructuralString... Fields>
decltype(auto) getFields(Tuple&& tuple, util::FieldList<Fields...>)
{
    if constexpr (sizeof...(Fields) == 1)
    {
        return std::get<getFieldIndex(Fields.value..., Dataset::fieldNames)>(std::forward<Tuple>(tuple));
    }
    else { return std::tie(std::get<getFieldIndex(Fields.value, Dataset::fieldNames)>(std::forward<Tuple>(tuple))...); }
}

template<util::StructuralString... Fields, class Dataset>
decltype(auto) getHost(Dataset& d)
{
    return getFields<Dataset>(d.dataTuple(), util::FieldList<Fields...>{});
}

template<class FL, class Dataset>
decltype(auto) getHost(Dataset& d)
{
    return getFields<Dataset>(d.dataTuple(), FL{});
}

template<util::StructuralString... Fields, class Dataset>
decltype(auto) getDevice(Dataset& d)
{
    return getFields<Dataset>(d.devData.dataTuple(), util::FieldList<Fields...>{});
}

template<class FL, class Dataset>
decltype(auto) getDevice(Dataset& d)
{
    return getFields<Dataset>(d.devData.dataTuple(), FL{});
}

template<class FL, class Dataset>
decltype(auto) get(Dataset& d)
{
    using AcceleratorType = typename Dataset::AcceleratorType;
    if constexpr (std::is_same_v<AcceleratorType, CpuTag>) { return getHost<FL>(d); }
    else { return getDevice<FL>(d); }
}

//! @brief Return a tuple of references to the specified particle field indices, to GPU fields if GPU is enabled
template<util::StructuralString... Fields, class Dataset>
decltype(auto) get(Dataset& d)
{
    return get<util::FieldList<Fields...>>(d);
}

//! @brief return a tuple of pointers to element i of @p tup = tuple of vector-like containers
template<class Tuple>
auto getPointers(Tuple&& tup, size_t i)
{
    return std::apply([i](auto&... tupEle) { return std::make_tuple(tupEle.data() + i...); }, std::forward<Tuple>(tup));
}

} // namespace cstone
