/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Utility functions to use compile-time strings as arguments to get on tuples
 *
 * Needs C++20 structural types
 */

#pragma once

#include "cstone/fields/data_util.hpp"
#include "cstone/primitives/primitives_acc.hpp"
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
