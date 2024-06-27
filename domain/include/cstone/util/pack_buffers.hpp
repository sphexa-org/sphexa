/*! @file
 * @brief Buffer description and management for domain decomposition particle exchanges
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/cuda/cuda_stubs.h"
#include "cstone/cuda/device_vector.h"
#include "cstone/primitives/gather.hpp"
#include "cstone/primitives/math.hpp"
#include "cstone/util/array.hpp"
#include "cstone/util/tuple_util.hpp"
#include "cstone/util/type_list.hpp"

namespace util
{

template<class... Arrays>
auto computeByteOffsets(size_t count, int alignment, Arrays... arrays)
{
    static_assert((... && std::is_pointer_v<Arrays>)&&"all arrays must be pointers");

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
using MakeL2ArrayPtrs = util::array<T*, 2>;

template<class T>
using PackType = std::conditional_t<sizeof(T) < sizeof(float), T, util::array<float, sizeof(T) / sizeof(float)>>;
} // namespace aux_traits

/*! @brief Compute multiple pointers such that the argument @p arrays can be mapped into a single buffer
 *
 * @tparam alignment        byte alignment of the individual arrays in the packed buffer
 * @tparam Arrays           arbitrary type of size 16 bytes or smaller
 * @param packedBufferBase  base address of the packed buffer
 * @param arraySize         number of elements of each @p array
 * @param arrays            independent array pointers
 * @return                  a tuple with (src, packed) pointers for each array
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
auto packBufferPtrs(char* packedBufferBase, size_t arraySize, Arrays... arrays)
{
    static_assert((... && std::is_pointer_v<Arrays>)&&"all arrays must be pointers");
    constexpr int numArrays = sizeof...(Arrays);
    constexpr util::array<size_t, numArrays> elementSizes{sizeof(std::decay_t<decltype(*arrays)>)...};
    constexpr auto indices = util::makeIntegralTuple(std::make_index_sequence<numArrays>{});

    const std::array<char*, numArrays> data{reinterpret_cast<char*>(arrays)...};

    auto arrayByteOffsets = computeByteOffsets(arraySize, alignment, arrays...);

    using Types     = util::TypeList<aux_traits::PackType<std::decay_t<decltype(*arrays)>>...>;
    using PtrTypes  = util::Map<aux_traits::MakeL2ArrayPtrs, Types>;
    using TupleType = util::Reduce<std::tuple, PtrTypes>;

    TupleType ret;

    auto packOneBuffer = [packedBufferBase, &data, &elementSizes, &arrayByteOffsets, &ret](auto index)
    {
        using ElementType = util::TypeListElement_t<index, Types>;
        auto* srcPtr      = reinterpret_cast<ElementType*>(data[index]);
        auto* packedPtr   = reinterpret_cast<ElementType*>(packedBufferBase + arrayByteOffsets[index]);

        std::get<index>(ret) = util::array<ElementType*, 2>{srcPtr, packedPtr};
    };

    util::for_each_tuple(packOneBuffer, indices);

    return ret;
}

//! calculate needed space in bytes
inline std::vector<size_t> computeByteOffsets(gsl::span<const size_t> numElements, int elementSize, int alignment)
{
    std::vector<size_t> ret(numElements.size() + 1, 0);
    for (int i = 0; i < numElements.ssize(); ++i)
    {
        ret[i] = cstone::round_up(numElements[i] * elementSize, alignment);
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
std::vector<gsl::span<T>> packAllocBuffer(Vector& vec, gsl::span<const size_t> numElements, int alignment)
{
    auto sizeBytes = computeByteOffsets(numElements, sizeof(T), alignment);
    reallocateBytes(vec, sizeBytes.back(), 1.0);

    std::vector<gsl::span<T>> ret(numElements.size());
    auto* basePtr = reinterpret_cast<char*>(rawPtr(vec));
    for (int i = 0; i < numElements.ssize(); ++i)
    {
        ret[i] = {reinterpret_cast<T*>(basePtr + sizeBytes[i]), numElements[i]};
    }
    return ret;
}

} // namespace util
