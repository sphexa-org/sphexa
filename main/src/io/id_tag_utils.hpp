/*
 * MIT License
 *
 * SPH-EXA
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * @brief  CPU/GPU Particle ID tag utilities
 *
 * @author Christopher Bignamini <christopher.bignamini@gmail.com>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <numeric>
#include <span>
#include <vector>

/*
 Remove the headers, template functions, and instantiate for IdType = uint64_t, LocalIndex = uint32_t
*/
#include "cstone/tree/definitions.h"
#include "sph/types.hpp"

namespace sphexa
{

using IdType = uint64_t;
using LocalIndex = uint32_t;
using CoordinateType = sph::SphTypes::CoordinateType;

/*! @brief Tagging mask definition (most significant bit flip)
 */
constexpr IdType msbMask = static_cast<IdType>(1) << (sizeof(IdType)*8 - 1);

/*! @brief Tagged id identification condition functor
 */
struct IsMasked
{
    HOST_DEVICE_FUN IdType operator()(IdType id) const
    {
        return (id & msbMask) != 0;
    }
};

/*! @brief Tagged id (in first:last range) identification, CPU version
 *
 * @param[in]  ids          ordered id list
 * @param[in]  first        first id index // TODO number of elements and pass iterator?
 * @param[in]  last         last (excluded) id index
 * @param[out] taggedIdsIndexes  vector of indexes of tagged ids
 */
void findTaggedIds(std::span<const IdType> ids, LocalIndex first, LocalIndex last, std::vector<LocalIndex>& taggedIdsIndexes);
//
// template<class IdTypeP, class LocalIndexP>
// void findTaggedIds(std::span<const IdTypeP> ids, LocalIndexP first, LocalIndexP last, std::vector<LocalIndexP>& taggedIdsIndexes);
//extern template void findTaggedIds(std::span<const uint64_t> ids, uint32_t first, uint32_t last, std::vector<uint32_t>& taggedIdsIndexes));
//void findTaggedIds(std::span<const IdType> ids, LocalIndex first, LocalIndex last, std::vector<LocalIndex>& taggedIdsIndexes);



// void findTaggedIds(std::span<const IdType> ids, LocalIndex first, LocalIndex last, std::vector<LocalIndex>& taggedIdsIndexes);

/*! @brief Tagged id (in first:last range) identification, GPU version
 *
 * @param[in]  ids          ordered id list
 * @param[in]  first        first id index // TODO number of elements and pass iterator?
 * @param[in]  last         last (excluded) id index
 * @param[out] taggedIdsIndexes  vector of indexes of tagged ids
 */
void findTaggedIdsGPU(std::span<const IdType> ids, LocalIndex first, LocalIndex last, std::vector<LocalIndex>& taggedIdsIndexes);

/*! @brief Id tagging (in first:last range) from list, CPU version
 *
 * @param[out] ids               id list
 * @param[in]  first             first id index // TODO number of elements and pass iterator?
 * @param[in]  last              last (excluded) id index
 * @param[in]  selectedIds       indexes to be tagged
 */
void tagIdsInList(std::span<IdType> ids, LocalIndex first, LocalIndex last, std::span<const IdType> selectedIds);


// Id tagging types selection
/*! @brief Id tagging spherical volume definition
 */
// TODO: selection spheres can be stored as cstone::Vec4<Tc>
struct IdSelectionSphere
{
    cstone::Vec3<CoordinateType> center;
    CoordinateType radius;
};
/*! @brief Id tagging list definition
 */
using IdSelectionList = std::vector<IdType>;


/*! @brief Id tagging (in first:last range) in spherical volume, CPU version
 *
 * @param[out] ids               id list
 * @param[in]  x                 x coordinates
 * @param[in]  y                 y coordinates
 * @param[in]  z                 z coordinates
 * @param[in]  first             first id index // TODO number of elements and pass iterator?
 * @param[in]  last              last (excluded) id index
 * @param[in]  selSphereData     spherical volume definition
 */
void tagIdsInSphere(std::span<IdType> ids, std::span<const CoordinateType> x, std::span<const CoordinateType> y,
    std::span<const CoordinateType> z, LocalIndex firstIndex, LocalIndex lastIndex, const IdSelectionSphere& selSphereData);

}