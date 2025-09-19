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

#include <vector>

// not needed
#include "cstone/cuda/device_vector.h"
/*
 Remove the headers, template functions, and instantiate for IdType = uint64_t, LocalIndex = uint32_t
*/
#include "cstone/tree/definitions.h"
#include "sph/particles_data.hpp"
#include "sph/types.hpp"
#include "ifile_io.hpp"

namespace sphexa
{

using IdType = uint64_t;//decltype(std::declval<ParticlesData<cstone::CpuTag>>().id)::value_type;// TODO: retrieve type from ParticlesData?
using IdVectorType = std::vector<IdType>;//decltype(std::declval<ParticlesData<cstone::CpuTag>>().id);
using CoordinateType = sph::SphTypes::CoordinateType;

/*! @brief Tagging mask definition (most significant bit flip)
 */
constexpr IdType msbMask = static_cast<IdType>(1) << (sizeof(IdType)*8 - 1);

/*! @brief Tagged id identification condition functor
 */
struct MaskFunctor
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
// template<class IdType, class LocalIndex>
// extern void findTaggedIds(std::span<const IdType> ids, LocalIndex first, LocalIndex last, std::vector<LocalIndex>& taggedIdsIndexes)
void findTaggedIds(const IdVectorType& ids, size_t first, size_t last, IdVectorType& taggedIdsIndexes);

/*! @brief Tagged id (in first:last range) identification, GPU version
 *
 * @param[in]  ids          ordered id list
 * @param[in]  first        first id index // TODO number of elements and pass iterator?
 * @param[in]  last         last (excluded) id index
 * @param[out] taggedIdsIndexes  vector of indexes of tagged ids
 */
void findTaggedIds(const cstone::DeviceVector<IdType>& ids, size_t first, size_t last, IdVectorType& taggedIdsIndexes);

/*! @brief Id tagging (in first:last range) from list, CPU version
 *
 * @param[out] ids               id list
 * @param[in]  first             first id index // TODO number of elements and pass iterator?
 * @param[in]  last              last (excluded) id index
 * @param[in]  selectedIds       indexes to be tagged
 */
void tagIdsInList(IdVectorType& ids, size_t first, size_t last, const IdVectorType& selectedIds);


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
using IdSelectionList = IdVectorType;


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
// TODO: all vectors can be spans or const-spans
void tagIdsInSphere(IdVectorType& ids, const std::vector<CoordinateType>& x, const std::vector<CoordinateType>& y,
    const std::vector<CoordinateType>& z, size_t firstIndex, size_t lastIndex, const IdSelectionSphere& selSphereData);

}