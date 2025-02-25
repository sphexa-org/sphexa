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

#include "cstone/cuda/device_vector.h"
#include "sph/types.hpp"

namespace sphexa
{

using ParticleIdType = uint64_t; // TODO: retrieve type from ParticlesData?
using CoordinateType  = sph::SphTypes::CoordinateType;

/*! @brief Tagging mask definition (most significant bit switch)
 */
constexpr ParticleIdType msbMask = static_cast<ParticleIdType>(1) << (sizeof(ParticleIdType)*8 - 1);

/*! @brief Tagged id (in first:last range) identification, CPU version
 *
 * @param[in]  ids          ordered id list
 * @param[in]  first        first id index // TODO number of elements and pass iterator?
 * @param[in]  last         last (excluded) id index
 * @param[out] taggedIdsIndexes  vector of indexes (positions wrt of selected particles)
 */
void findTaggedIds(const std::vector<uint64_t>& ids, size_t first, size_t last, std::vector<uint64_t>& taggedIdsIndexes);

/*! @brief Tagged id (in first:last range) identification, GPU version
 *
 * @param[in]  ids          ordered id list
 * @param[in]  first        first id index // TODO number of elements and pass iterator?
 * @param[in]  last         last (excluded) id index
 * @param[out] taggedIdsIndexes  vector of indexes (positions wrt of selected particles)
 */
void findTaggedIds(const cstone::DeviceVector<uint64_t>& ids, size_t first, size_t last, std::vector<uint64_t>& taggedIdsIndexes);

/*! @brief Id tagging (in first:last range) from list, CPU version
 *
 * @param[out] ids          ordered id list
 * @param[in]  first        first id index // TODO number of elements and pass iterator?
 * @param[in]  last         last (excluded) id index
 * @param[in]  selectedIds  indexes to be tagged
 */
void tagIdsInList(std::vector<uint64_t>& ids, size_t first, size_t last, const std::vector<uint64_t>& selectedIds);

/*! @brief Id tagging (in first:last range) from list, GPU version
 *
 * @param[out] ids          ordered id list
 * @param[in]  first        first id index // TODO number of elements and pass iterator?
 * @param[in]  last         last (excluded) id index
 * @param[in]  selectedIds  indexes to be tagged
 */
void tagIdsInList(cstone::DeviceVector<uint64_t>& ids, size_t first, size_t last, const std::vector<uint64_t>& selectedIds);

/*! @brief Id tagging spherical volume definition
 */
struct IdSelectionSphere
{
    std::array<double, 3> center;
    double radius;
};

/*! @brief Id tagging (in first:last range) in spherical volume, CPU version
 *
 * @param[out] ids           ordered id list
 * @param[in]  x             x coordinates
 * @param[in]  y             y coordinates
 * @param[in]  z             z coordinates
 * @param[in]  first         first id index // TODO number of elements and pass iterator?
 * @param[in]  last          last (excluded) id index
 * @param[in]  selSphereData spherical volume definition
 */
void tagIdsInSphere(std::vector<uint64_t>& ids, const std::vector<CoordinateType>& x, const std::vector<CoordinateType>& y,
    const std::vector<CoordinateType>& z, size_t firstIndex, size_t lastIndex, const IdSelectionSphere& selSphereData);

/*! @brief Id tagging (in first:last range) in spherical volume, GPU version
 *
 * @param[out] ids           ordered id list
 * @param[in]  x             x coordinates
 * @param[in]  y             y coordinates
 * @param[in]  z             z coordinates
 * @param[in]  first         first id index // TODO number of elements and pass iterator?
 * @param[in]  last          last (excluded) id index
 * @param[in]  selSphereData spherical volume definition
 */
void tagIdsInSphere(cstone::DeviceVector<uint64_t>& ids, const std::vector<CoordinateType>& x, const std::vector<CoordinateType>& y,
    const std::vector<CoordinateType>& z, size_t firstIndex, size_t lastIndex, const IdSelectionSphere& selSphereData);


}