/*
 * Cornerstone octree
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: MIT License
 */

/*! @file
 * @brief Buffer description and management for domain decomposition particle exchanges
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <numeric>
#include <span>
#include <vector>

#include "cstone/primitives/math.hpp"
#include "cstone/util/array.hpp"
#include "cstone/util/reallocate.hpp"
#include "cstone/util/tuple_util.hpp"
#include "cstone/util/type_list.hpp"

namespace util
{

template<class... Arrays>
auto computeByteOffsets(size_t count, int alignment, Arrays... arrays)
{
    static_assert((... && std::is_pointer_v<Arrays>), "all arrays must be pointers");

    util::array<size_t, sizeof...(Arrays) + 1> byteOffsets{sizeof(std::decay_t<decltype(*arrays)>)...};
    byteOffsets *= count;

    //! each sub-buffer will be aligned on a @a alignment-byte aligned boundary
    for (size_t i = 0; i < byteOffsets.size(); ++i)
    {
        byteOffsets[i] = cstone::round_up(byteOffsets[i], alignment);
    }

    std::exclusive_scan(byteOffsets.begin(), byteOffsets.end(), byteOffsets.begin(), size_t(0));

    return byteOffsets;
}

namespace aux_traits
{
template<class T>
using PackType = std::conditional_t<sizeof(T) < sizeof(float), T, util::array<float, sizeof(T) / sizeof(float)>>;
} // namespace aux_traits

/*! @brief Compute multiple pointers such that the argument @p arrays can be mapped into a single buffer
 *
 * @tparam alignment    byte alignment of the individual arrays in the packed buffer
 * @tparam Arrays       arbitrary type of size 16 bytes or smaller
 * @param base          base address of the packed buffer
 * @param arraySize     number of elements of each @p array
 * @param arrays        independent array pointers
 * @return              a tuple with (src, packed) pointers for each array
 *
 * @p arrays:    ------      ------         ------
 *   (pointers)  a           b              c
 *
 * packedBuffer    |------|------|------|
 *  (pointer)       A      B      C
 *                  |
 *                  packedBufferBase
 *
 * return  tuple( (a, A), (b, B), (c, C) )
 *
 * Pointer types are util::array<float, sizeof(*(a, b or c) / sizeof(float)>..., i.e. the same size as the original
 * element types. This is done to express all types up to 16 bytes with just four util::array types in order
 * to reduce the number of gather/scatter GPU kernel template instantiations.
 */
template<int alignment, class... Arrays>
auto packBufferPtrs(void* base, size_t arraySize, Arrays... arrays)
{
    static_assert((... && std::is_pointer_v<Arrays>), "all arrays must be pointers");
    auto arrayByteOffsets = computeByteOffsets(arraySize, alignment, arrays...);
    using Types           = TypeList<aux_traits::PackType<std::decay_t<decltype(*arrays)>>...>;

    auto packBuffers = [base, &arrayByteOffsets]<std::size_t... Is>(std::index_sequence<Is...>, auto data)
    {
        auto* base_c = reinterpret_cast<char*>(base);
        return std::make_tuple(util::array<TypeListElement_t<Is, Types>*, 2>{
            reinterpret_cast<TypeListElement_t<Is, Types>*>(std::get<Is>(data)),
            reinterpret_cast<TypeListElement_t<Is, Types>*>(base_c + arrayByteOffsets[Is])}...);
    };

    return packBuffers(std::make_index_sequence<sizeof...(Arrays)>{}, std::make_tuple(arrays...));
}

//! calculate needed space in bytes
inline std::vector<size_t> computeByteOffsets(std::span<const size_t> numElements, int elementSize, int alignment)
{
    std::vector<size_t> ret(numElements.size() + 1, 0);
    for (std::size_t i = 0; i < numElements.size(); ++i)
    {
        ret[i] = cstone::round_up(numElements[i] * elementSize, alignment);
    }
    std::exclusive_scan(ret.begin(), ret.end(), ret.begin(), size_t(0));
    return ret;
}

//! calculate needed space in bytes
template<std::size_t N>
constexpr std::array<size_t, N + 1>
computeByteOffsets(std::array<size_t, N> numElements, std::array<std::size_t, N> elementSizes, int alignment)
{
    std::array<size_t, N + 1> ret;
    for (std::size_t i = 0; i < numElements.size(); ++i)
    {
        ret[i] = cstone::round_up(numElements[i] * elementSizes[i], alignment);
    }
    std::exclusive_scan(ret.begin(), ret.end(), ret.begin(), size_t(0));
    return ret;
}

/*! @brief allocate space for sum(numElements) elements and return pointers to each subrange
 *
 * @param[inout]  vec          vector-like container with linear memory
 * @param[in]     numElements  sequence of subrange sizes
 * @param[in]     alignment    subrange alignment requirement
 * @return                     a vector with a pointer into @p vec for each subrange
 */
template<class T, class Vector>
std::vector<std::span<T>> packAllocBuffer(Vector& vec, std::span<const size_t> numElements, int alignment)
{
    auto sizeBytes = computeByteOffsets(numElements, sizeof(T), alignment);
    reallocateBytes(vec, sizeBytes.back(), 1.0);

    std::vector<std::span<T>> ret(numElements.size());
    auto* basePtr = reinterpret_cast<char*>(vec.data());
    for (std::size_t i = 0; i < numElements.size(); ++i)
    {
        ret[i] = {reinterpret_cast<T*>(basePtr + sizeBytes[i]), numElements[i]};
    }
    return ret;
}

template<class Vector, class... Ts>
std::tuple<std::span<Ts>...>
packAllocBuffer(Vector& vec, TypeList<Ts...>, std::array<std::size_t, sizeof...(Ts)> numElements, int alignment)
{
    std::array<std::size_t, sizeof...(Ts)> elementSizes{sizeof(Ts)...};
    auto offsets = computeByteOffsets(numElements, elementSizes, alignment);

    reallocateBytes(vec, offsets.back(), 1.0);
    char* basePtr = reinterpret_cast<char*>(vec.data());

    auto packBuffers = [basePtr, &offsets, &numElements]<std::size_t... Is>(std::index_sequence<Is...>)
    { return std::make_tuple(std::span<Ts>{reinterpret_cast<Ts*>(basePtr + offsets[Is]), numElements[Is]}...); };

    return packBuffers(std::make_index_sequence<sizeof...(Ts)>{});
}

} // namespace util
